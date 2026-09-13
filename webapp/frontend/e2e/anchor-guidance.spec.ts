import {test, expect, type Page} from '@playwright/test';
import path from 'node:path';

async function state(page: Page) { return (await page.request.get('/api/state')).json(); }
async function setup(page: Page, masks = true) {
  await page.request.post('/api/reset');
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles([1,2,3].map(i => path.resolve('e2e/fixtures', `photo-${i}.png`)));
  await expect(page.locator('.photo-card')).toHaveCount(3);
  const s = await state(page), fid = s.images[0].id, mid = s.images[1].id;
  if (masks) {
    for (const id of [fid, mid]) {
      expect((await page.request.post(`/api/mask/${id}/click`, {data:{x:300,y:300,label:1}})).ok()).toBeTruthy();
      expect((await page.request.post(`/api/mask/${id}/action`, {data:{action:'confirm'}})).ok()).toBeTruthy();
    }
  }
  await page.getByRole('button', {name:'현재 정합',exact:true}).click();
  await expect.poll(async () => !!(await state(page)).images[1].result).toBe(true);
  await expect(page.getByRole('button', {name:'현재 정합',exact:true})).toBeEnabled();
  return {fid,mid};
}
const panel = (page:Page) => page.getByRole('region', {name:'정합 안내'});

test('persistent guidance recommends editable draft pairs, reruns similarity and restores previous result', async ({page}) => {
  const {fid,mid} = await setup(page);
  const before = (await state(page)).images[1].result.id;
  const dialogs: string[] = [];
  page.on('dialog', dialog => {dialogs.push(dialog.message()); void dialog.dismiss();});
  await expect(panel(page)).toContainText('정합 결과가 마음에 들지 않으면');
  await expect(panel(page)).toContainText('마스크를 모두 합쳐');
  await expect(panel(page).locator('.recovery-actions button')).toHaveCount(4);
  const response = page.waitForResponse(r => r.url().endsWith(`/api/anchors/${mid}/recommend`));
  await page.getByRole('button', {name:'앵커 자동 추천',exact:true}).click();
  const count = (await (await response).json()).pairs.length;
  expect(count).toBeGreaterThanOrEqual(4);
  expect(count).toBeLessThanOrEqual(24);
  await expect(panel(page)).toContainText(`추천 앵커 ${count}쌍`);
  await expect(page.locator('[data-anchor^="auto-"]')).toHaveCount(count*2);
  expect((await (await page.request.get(`/api/anchors/${mid}`)).json()).pairs).toHaveLength(0);
  expect((await state(page)).images[1].result.id).toBe(before);
  await page.locator('.anchor-list button').first().click();
  await page.getByRole('button', {name:'선택/입력 취소 D',exact:true}).click();
  await expect(page.locator('[data-anchor^="auto-"]')).toHaveCount((count-1)*2);
  await page.getByRole('button', {name:'이 점들로 재정합',exact:true}).click();
  await expect.poll(async () => (await state(page)).images[1].result.gate).toBe('anchor_similarity');
  await expect(page.getByRole('button', {name:'현재 정합',exact:true})).toBeEnabled();
  const anchors = (await (await page.request.get(`/api/anchors/${mid}`)).json()).pairs;
  expect(anchors).toHaveLength(count-1);
  expect((await state(page)).images[1].result.fixed_id).toBe(fid);
  await page.getByRole('button', {name:'와이프',exact:true}).click();
  await page.getByRole('button', {name:'이전 정합 결과 비교',exact:true}).click();
  await expect(page.getByRole('slider', {name:'와이프 경계 이동',exact:true})).toBeVisible();
  await page.getByRole('button', {name:'이전 결과로 돌아가기',exact:true}).click();
  await expect.poll(async () => (await state(page)).images[1].result.id).toBe(before);
  await expect(page.getByRole('button', {name:'앵커 자동 추천',exact:true})).toBeEnabled();
  await expect(page.getByRole('button', {name:'이전 정합 결과 비교',exact:true})).toBeVisible();
  expect(dialogs).toEqual([]);
  await page.screenshot({path:'test-results/anchor-guidance-desktop.png'});
});

test('recommended pairs can be moved before applying and cancel restores saved manual pairs', async ({page}) => {
  const {fid,mid} = await setup(page);
  const manual = {id:'manual-user', fixed:[200,200], moving:[205,203], enabled:true};
  await page.request.put(`/api/anchors/${mid}`, {data:{fixed_id:fid, base_revision:0, pairs:[manual]}});
  const response = page.waitForResponse(r => r.url().endsWith(`/api/anchors/${mid}/recommend`));
  await page.getByRole('button', {name:'앵커 자동 추천',exact:true}).click();
  const count = (await (await response).json()).pairs.length;
  expect(count).toBeGreaterThanOrEqual(4);
  await expect(page.locator('[data-anchor^="auto-"]')).toHaveCount(count*2);
  await expect(page.locator('[data-anchor="manual-user"]')).toHaveCount(2);
  // Dense recommendations may overlap at fit-to-window scale. Selecting a
  // pair in the list must bring that marker above the other recommendations.
  await page.locator('.anchor-list button').nth(1).click();
  const point = page.locator('[data-anchor^="auto-"]').first();
  await expect(point).toHaveClass(/selected/);
  await point.scrollIntoViewIfNeeded();
  const box = (await point.boundingBox())!;
  await page.mouse.move(box.x+box.width/2, box.y+box.height/2);
  await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+12, box.y+box.height/2+6, {steps:3});
  await page.mouse.up();
  const moved = (await point.boundingBox())!;
  expect(moved.x-box.x).toBeGreaterThan(8);
  expect((await (await page.request.get(`/api/anchors/${mid}`)).json()).pairs).toHaveLength(1);
  await page.screenshot({path:'test-results/anchor-recommendation-preview.png'});
  await page.getByRole('button', {name:'추천 취소',exact:true}).click();
  await expect(page.locator('[data-anchor^="auto-"]')).toHaveCount(0);
  await expect(page.locator('[data-anchor="manual-user"]')).toHaveCount(2);
});

test('four recovery choices navigate without erasing masks; guide survives hidden inspector and narrow focus mode', async ({page}) => {
  const {mid} = await setup(page);
  const before = (await state(page)).images[1].n_objects;
  await panel(page).getByRole('button', {name:'마스크 재선택',exact:true}).click();
  expect((await state(page)).images[1].n_objects).toBe(before);
  await expect(page.getByRole('button', {name:'현재 사진',exact:true})).toBeVisible();
  await page.getByRole('button', {name:'앵커 직접 찍기',exact:true}).click();
  await expect(page.locator('#anchor-controls')).toBeVisible();
  await expect(panel(page)).toContainText('기준 사진의 한 점');
  await panel(page).getByRole('button', {name:'미세조정',exact:true}).click();
  await expect(page.locator('.adjustment')).toBeVisible();
  await page.getByRole('button', {name:'비교',exact:true}).click();
  await page.getByRole('button', {name:'집중 보기 F',exact:true}).click();
  await expect(panel(page)).toBeVisible();
  await page.setViewportSize({width:600,height:850});
  await panel(page).scrollIntoViewIfNeeded();
  const buttons = panel(page).locator('.recovery-actions button');
  const boxes = await Promise.all([0,1,2,3].map(i => buttons.nth(i).boundingBox()));
  expect(boxes[0]!.y).toBe(boxes[1]!.y);
  expect(boxes[2]!.y).toBeGreaterThan(boxes[0]!.y);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
  await page.screenshot({path:'test-results/anchor-guidance-narrow.png'});
  expect((await (await page.request.get(`/api/anchors/${mid}`)).json()).pairs).toHaveLength(0);
});

test('late recommendation cannot attach points or switch tools on another photo', async ({page}) => {
  await setup(page);
  let release!: () => void;
  const wait = new Promise<void>(resolve => {release = resolve;});
  await page.route('**/api/anchors/*/recommend', async route => {
    const response = await route.fetch();
    await wait;
    await route.fulfill({response});
  });
  await page.getByRole('button', {name:'앵커 자동 추천',exact:true}).click();
  await page.getByRole('button', {name:'사진 보기 photo-3.png',exact:true}).click();
  release();
  await expect(page.getByRole('button', {name:'앵커 자동 추천',exact:true})).toBeEnabled();
  await expect(page.getByTestId('active-name')).toHaveText('photo-3.png');
  await expect(page.locator('[data-anchor^="auto-"]')).toHaveCount(0);
  await expect(panel(page)).not.toContainText('추천 앵커');
});

test('missing masks gives inline actionable message without replacing result', async ({page}) => {
  await setup(page, false);
  const before = (await state(page)).images[1].result.id;
  await page.getByRole('button', {name:'앵커 자동 추천',exact:true}).click();
  await expect(panel(page)).toContainText('Z로 확정하세요');
  expect((await state(page)).images[1].result.id).toBe(before);
});

test('mask action is first and the default follows mask availability', async ({page}) => {
  await setup(page, false);
  const actions = panel(page).locator('.recovery-actions button');
  await expect(actions).toHaveText(['마스크 선택', '앵커 자동 추천', '앵커 직접 찍기', '미세조정']);
  await expect(actions.nth(0)).toHaveClass(/recommended/);
  await expect(actions.nth(1)).not.toHaveClass(/recommended/);
  await expect(panel(page).locator('#mask-help')).toBeVisible();
  await setup(page, true);
  await expect(actions).toHaveText(['마스크 재선택', '앵커 자동 추천', '앵커 직접 찍기', '미세조정']);
  await expect(actions.nth(0)).not.toHaveClass(/recommended/);
  await expect(actions.nth(1)).toHaveClass(/recommended/);
  await expect(panel(page).locator('#recommend-help')).toBeVisible();
  await actions.nth(0).hover();
  await expect(panel(page).locator('#mask-help')).toBeVisible();
  await expect(panel(page).locator('#recommend-help')).toBeHidden();
});

test('manual anchor workflow persists across photos while pairs stay photo-specific', async ({page}) => {
  const {fid, mid} = await setup(page, false);
  await page.request.put(`/api/anchors/${mid}`, {data:{fixed_id:fid, base_revision:0,
    pairs:[{id:'saved-second',fixed:[200,200],moving:[205,203],enabled:true}]}});
  await panel(page).getByRole('button', {name:'앵커 직접 찍기',exact:true}).click();
  await page.getByRole('button', {name:'사진 보기 photo-3.png',exact:true}).click();
  await expect(page.locator('#anchor-controls')).toBeVisible();
  await expect(page.locator('#anchor-controls .instruction')).toContainText('기준 사진을 클릭하고 A로 선택하세요');
  await expect(panel(page)).toContainText('기준 사진의 한 점');
  await expect(page.locator('[data-anchor="saved-second"]')).toHaveCount(0);
  await page.getByRole('button', {name:'사진 보기 photo-2.png',exact:true}).click();
  await expect(page.locator('[data-anchor="saved-second"]')).toHaveCount(2);
  await expect(page.locator('#anchor-controls .instruction')).toContainText('기준 사진을 클릭하고 A로 선택하세요');
  await panel(page).getByRole('button', {name:'마스크 선택',exact:true}).click();
  await page.getByRole('button', {name:'사진 보기 photo-3.png',exact:true}).click();
  await expect(panel(page)).not.toContainText('기준 사진의 한 점');
});
