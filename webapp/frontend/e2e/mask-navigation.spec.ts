import { test, expect } from '@playwright/test';
import path from 'node:path';

test.beforeEach(async ({page}) => {
  await page.request.post('/api/reset');
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles([1,2,3].map(i => path.resolve('e2e/fixtures', `photo-${i}.png`)));
  await expect(page.locator('.photo-card')).toHaveCount(3);
});

test('mask first: anchor controls and dots are hidden until expanded; loupe removed', async ({page}) => {
  await expect(page.getByRole('button', {name:'대응점 선택 A',exact:true})).not.toBeVisible();
  await expect(page.getByRole('button', {name:'부분 확대',exact:true})).toHaveCount(0);
  const pane = page.locator('.photo-viewport').last();
  await pane.click({position:{x:180,y:160}});
  await expect(page.getByRole('button',{name:'개체 확정 Z',exact:true})).toBeEnabled();
  await expect(page.locator('[data-anchor]')).toHaveCount(0);
  await page.getByRole('button',{name:'대응점 펼치기',exact:true}).click();
  await expect(page.getByRole('button',{name:'대응점 선택 A',exact:true})).toBeVisible();
  await expect(page.locator('[data-anchor="candidate"]')).toBeVisible();
  await page.getByRole('button',{name:'대응점 접기',exact:true}).click();
  await expect(page.locator('[data-anchor]')).toHaveCount(0);
  await page.keyboard.press('a');
  await expect(page.locator('[data-anchor]')).toHaveCount(0);
});

test('switching masked photos never displays the previous overlay while the next loads', async ({page}) => {
  const s = await (await page.request.get('/api/state')).json();
  for (const image of s.images.slice(1)) {
    await page.request.post(`/api/mask/${image.id}/click`, {data:{x:180,y:200,label:1}});
    await page.request.post(`/api/mask/${image.id}/action`, {data:{action:'confirm'}});
  }
  await page.reload();
  await page.getByRole('button',{name:'사진 보기 photo-2.png',exact:true}).click();
  const pane = page.locator('.photo-viewport').last();
  await expect.poll(() => pane.locator('img').evaluateAll(imgs => imgs.some(i => i.currentSrc.includes('/overlay') && i.complete))).toBe(true);
  const oldLayer = await pane.locator('img[src*="/overlay"]').elementHandle();
  let release!: () => void;
  const hold = new Promise<void>(r => release = r);
  await page.route(`**/api/mask/${s.images[2].id}/overlay?*`, async route => { await hold; await route.continue(); });
  try {
    await page.getByRole('button',{name:'사진 보기 photo-3.png',exact:true}).click();
    await expect(pane).toHaveAttribute('aria-label', /photo-3/);
    expect(await oldLayer!.evaluate(el => el.isConnected)).toBe(false);
    const old = `/api/mask/${s.images[1].id}/overlay`;
    await expect.poll(() => pane.locator('img').evaluateAll((imgs, old) => imgs.some(i => i.currentSrc.includes(old)), old), {timeout:1200}).toBe(false);
  } finally { release(); }
});

test('rapid mask clicks keep one active request and only the latest queued preview', async ({page}) => {
  let release!: () => void;
  const hold = new Promise<void>(r => release = r);
  const requests: any[] = [];
  await page.route('**/api/mask/*/preview', async route => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) await hold;
    await route.continue();
  });
  const pane = page.locator('.photo-viewport').last();
  try {
    await pane.click({position:{x:150,y:160}});
    await expect.poll(() => requests.length).toBe(1);
    for (let i=0;i<6;i++) await pane.click({position:{x:180+i*12,y:170}});
    await page.waitForTimeout(150); // Deliberately hold inference to expose queued requests.
    expect(requests).toHaveLength(1);
  } finally { release(); }
  await expect(page.getByRole('button',{name:'개체 확정 Z',exact:true})).toBeEnabled();
  expect(requests).toHaveLength(2);
  expect(requests[1].points).toHaveLength(7);
});

test('Ctrl Z immediately after Z waits for commit response then undoes the saved mask', async ({page}) => {
  const state = await (await page.request.get('/api/state')).json();
  const id = state.images[1].id;
  const pane = page.locator('.photo-viewport').last();
  await pane.click({position:{x:180,y:160}});
  await expect(page.getByRole('button',{name:'개체 확정 Z',exact:true})).toBeEnabled();
  let release!: () => void;
  const hold = new Promise<void>(r => release = r);
  let committed = false;
  await page.route(`**/api/mask/${id}/action`, async route => {
    const response = await route.fetch();
    committed = true;
    await hold;
    await route.fulfill({response});
  });
  let undos = 0;
  page.on('request', r => { if (r.url().endsWith('/api/history/undo')) undos++; });
  try {
    await page.keyboard.press('z');
    await expect.poll(() => committed).toBe(true);
    await page.keyboard.press('Control+z');
    await page.waitForTimeout(100);
    expect(undos).toBe(0);
  } finally { release(); }
  await expect.poll(async () => (await (await page.request.get('/api/state')).json()).images[1].n_objects).toBe(0);
  expect(undos).toBe(1);
});
