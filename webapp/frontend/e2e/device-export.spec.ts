import {test, expect} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';

test.beforeEach(async ({page}) => {
  await page.request.post('/api/reset');
  await page.goto('/');
});

test('download exports JPG and ZIP with visible progress and no folder picker', async ({page}) => {
  let pickerCalls = 0;
  await page.route('**/api/select_folder', route => { pickerCalls++; return route.fulfill({json:{path:null}}); });
  await page.locator('input[type=file]').setInputFiles([1,2,3].map(i=>path.resolve(`e2e/fixtures/photo-${i}.png`)));
  await expect(page.locator('.photo-card')).toHaveCount(3);
  await page.getByRole('button',{name:'전체 정합',exact:true}).click();
  await expect(page.getByRole('button',{name:'현재 정합',exact:true})).toBeEnabled();
  const save = page.getByRole('button',{name:'현재 결과 저장',exact:true});
  await expect(save).toBeEnabled();
  let release!:()=>void;
  const barrier = new Promise<void>(resolve=>release=resolve);
  await page.route('**/api/export',async route=>{ await barrier; await route.continue(); });
  const jpgEvent = page.waitForEvent('download');
  try {
    await save.click();
    await expect(save).toBeDisabled();
    await expect(page.locator('.export-notice')).toContainText('준비');
  } finally { release(); }
  const jpg = await jpgEvent;
  expect(jpg.suggestedFilename()).toMatch(/\.jpg$/);
  expect((await fs.readFile((await jpg.path())!)).subarray(0,2).toString('hex')).toBe('ffd8');
  await expect(page.locator('.export-notice')).toContainText('다운로드를 요청');
  await page.unroute('**/api/export');
  const zipEvent = page.waitForEvent('download');
  await page.getByRole('button',{name:'전체 결과 저장',exact:true}).click();
  const zip = await zipEvent;
  expect(zip.suggestedFilename()).toMatch(/\.zip$/);
  expect((await fs.readFile((await zip.path())!)).subarray(0,2).toString()).toBe('PK');
  expect(pickerCalls).toBe(0);

  await page.route('**/api/export',route=>route.fulfill({status:503,json:{detail:'테스트 저장 실패'}}));
  await save.click();
  await expect(page.locator('.export-notice')).toContainText('테스트 저장 실패');
  await expect(save).toBeEnabled();
  await page.getByLabel('저장 방식').selectOption('folder');
  await save.click();
  await expect(page.locator('.export-notice')).toContainText('폴더');
  await page.getByRole('button',{name:'폴더 찾기',exact:true}).click();
  await expect(page.locator('.export-notice')).toContainText('취소');
});

test('settings explain CPU-only state and switch both accelerators through API', async ({page}) => {
  const gpu:any={mode:'auto',device:'cpu',accelerator:null,platform:'win32',gpu_name:'NVIDIA test',frozen:true,installed:false,models:{}};
  await page.route('**/api/gpu', route=>route.fulfill({json:gpu}));
  await page.reload();
  await page.getByText('정합 설정',{exact:true}).click();
  await expect(page.getByTestId('cpu-help')).toContainText('설치');
  await expect(page.getByRole('button',{name:'CPU',exact:true})).toBeHidden();
  await page.locator('.device-disclosure > summary').click();
  await expect(page.getByRole('button',{name:'GPU 가속',exact:true})).toBeDisabled();
  await expect(page.getByText('(테스트용)',{exact:true})).toBeVisible();
  expect(await page.locator('option').filter({hasText:'엄격'}).count()).toBe(0);
  for (const accelerator of ['cuda','mps']) {
    gpu.accelerator=accelerator; gpu.device=accelerator;
    gpu.platform=accelerator==='mps'?'darwin':'win32';
    await page.reload();
    await page.getByText('정합 설정',{exact:true}).click();
    await expect(page.getByRole('button',{name:'CPU',exact:true})).toBeHidden();
    await page.locator('.device-disclosure > summary').click();
    const modes:string[]=[];
    await page.route('**/api/gpu/device',route=>{
      const mode=route.request().postDataJSON().mode; modes.push(mode);
      gpu.mode=mode; gpu.device=mode==='cpu'?'cpu':accelerator;
      return route.fulfill({json:gpu});
    });
    await page.getByRole('button',{name:'CPU',exact:true}).click();
    await expect(page.getByTestId('cpu-help')).toContainText('테스트');
    await expect(page.getByRole('button',{name:'CPU',exact:true})).toHaveAttribute('aria-pressed','true');
    await page.getByRole('button',{name:'GPU 가속',exact:true}).click();
    await expect(page.getByRole('button',{name:'GPU 가속',exact:true})).toHaveAttribute('aria-pressed','true');
    expect(modes).toEqual(['cpu','auto']);
    await page.locator('.device-disclosure > summary').click();
    await expect(page.getByRole('button',{name:'CPU',exact:true})).toBeHidden();
    await page.unroute('**/api/gpu/device');
  }
});
