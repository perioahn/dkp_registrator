import { test, expect } from '@playwright/test';
import path from 'node:path';

test.beforeEach(async ({page}) => {
  await page.request.post('/api/reset');
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles([1,2,3].map(i => path.resolve('e2e/fixtures', `photo-${i}.png`)));
  await expect(page.locator('.photo-card')).toHaveCount(3);
});

test('focus enlarges photos and restores sidebar preference without resetting navigation', async ({page}) => {
  const area = page.locator('.comparison-area');
  const before = await area.boundingBox();
  await page.getByRole('button', {name:'왼쪽 목록',exact:true}).click();
  await page.getByRole('button', {name:'사진 보기 photo-2.png',exact:true}).click();
  await page.keyboard.press('f');
  await expect(page.locator('.app')).toHaveClass(/focus-mode/);
  await expect(page.locator('.photo-browser')).toBeHidden();
  await expect(page.locator('.inspector')).toBeHidden();
  await expect.poll(async () => {
    const now = await area.boundingBox();
    return now!.width * now!.height / (before!.width * before!.height);
  }).toBeGreaterThan(1.3);
  await page.keyboard.press('ArrowRight');
  await expect(page.getByTestId('active-name')).toHaveText('photo-3.png');
  await expect(page.locator('.photo-viewport').last()).toHaveAttribute('aria-busy','false');
  const image = page.locator('.photo-viewport').last().locator('img').first();
  await expect.poll(() => image.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
  await image.evaluate((img: HTMLImageElement) => img.decode());
  const imageBox = await image.boundingBox();
  expect(imageBox!.width).toBeGreaterThan(500);
  await page.screenshot({path:'test-results/focus-desktop.png'});
  await page.keyboard.press('Escape');
  await expect(page.locator('.app')).not.toHaveClass(/focus-mode/);
  await expect(page.locator('.app')).toHaveClass(/list-side/);
  await expect(page.locator('.photo-browser')).toBeVisible();
  await expect(page.locator('.inspector')).toBeVisible();
  await expect(page.getByTestId('active-name')).toHaveText('photo-3.png');
  await page.getByRole('searchbox', {name:'파일명 검색'}).fill('f');
  await expect(page.locator('.app')).not.toHaveClass(/focus-mode/);
});

test('compact screen keeps version visible and thumbnails request only 256 pixels', async ({page}) => {
  await page.setViewportSize({width:1024,height:600});
  await expect(page.getByTestId('app-version')).toBeVisible();
  await expect(page.getByTestId('app-version')).toContainText(/1\.5\.1-local\.\d+/);
  const src = await page.locator('.thumbnail-button img').first().getAttribute('src');
  expect(src).toContain('max_side=256');
  await expect(page.locator('.photo-viewport.mask-active')).toHaveCount(1);
  const area = await page.locator('.comparison-area').boundingBox();
  expect(area!.height).toBeGreaterThan(145);
  await page.screenshot({path:'test-results/compact-workspace.png'});
});

test('reconnect refreshes session and bursts of focus events coalesce state reads', async ({page}) => {
  await page.addInitScript(() => {
    const Native = window.EventSource;
    window.EventSource = class extends Native {
      constructor(url: string | URL, options?: EventSourceInit) {
        super(url, options);
        (window as any).testEvents = this;
      }
    };
  });
  await page.reload();
  await expect(page.getByTestId('active-name')).toBeVisible();
  await page.evaluate(() => (window as any).testEvents.dispatchEvent(new Event('error')));
  await expect(page.locator('.connection-warning')).toBeVisible();
  let release!: () => void;
  const hold = new Promise<void>(r => release = r);
  let requests = 0;
  await page.route('**/api/state', async route => {
    requests++;
    if (requests === 1) await hold;
    await route.continue();
  });
  try {
    await page.evaluate(() => (window as any).testEvents.dispatchEvent(new Event('open')));
    await expect.poll(() => requests).toBe(1);
    await page.evaluate(() => { for (let i=0;i<10;i++) window.dispatchEvent(new Event('focus')); });
    await page.waitForTimeout(100);
    expect(requests).toBe(1);
  } finally { release(); }
  await expect.poll(() => requests).toBe(2);
  await expect(page.locator('.connection-warning')).toBeHidden();
});

test('switching photos shows loading state while new image is delayed', async ({page}) => {
  const state = await (await page.request.get('/api/state')).json();
  let release!: () => void;
  const hold = new Promise<void>(r => release = r);
  await page.route(`**/api/image/${state.images[2].id}?*`, async route => {
    if (!route.request().url().includes('max_side=256')) await hold;
    await route.continue();
  });
  try {
    await page.getByRole('button', {name:'사진 보기 photo-3.png',exact:true}).click();
    await expect(page.locator('.photo-viewport').last()).toHaveAttribute('aria-busy','true');
    await expect(page.locator('.viewport-loading').last()).toBeVisible();
  } finally { release(); }
  await expect(page.locator('.photo-viewport').last()).toHaveAttribute('aria-busy','false');
});
