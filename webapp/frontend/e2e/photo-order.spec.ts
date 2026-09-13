import {test, expect} from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';

test.beforeEach(async ({page}) => {
  await page.request.post('/api/reset');
  await page.goto('/');
  await page.locator('input[type=file]').setInputFiles([1,2,3].map(i => path.resolve('e2e/fixtures',`photo-${i}.png`)));
  await expect(page.locator('.photo-card')).toHaveCount(3);
});

test('thumbnail drag reorders without uploading and survives refresh', async ({page}) => {
  const state = await (await page.request.get('/api/state')).json();
  let uploads = 0;
  page.on('request',r => { if(r.method()==='POST' && r.url().endsWith('/api/upload')) uploads++; });
  const cards = page.locator('.photo-card');
  await cards.nth(2).locator('img').dragTo(cards.first(), {targetPosition:{x:8,y:25}});
  await expect(cards.first().locator('img')).toHaveAttribute('alt','photo-3.png');
  await page.reload();
  await expect(cards.first().locator('img')).toHaveAttribute('alt','photo-3.png');
  expect(uploads).toBe(0);
  const after = await (await page.request.get('/api/state')).json();
  expect(after.images.map((p:any)=>p.id)).toEqual(state.images.map((p:any)=>p.id));
  expect(after.fixed_id).toBe(state.fixed_id);
  await cards.first().dragTo(page.getByTestId('fixed-slot'));
  await expect(page.getByTestId('fixed-slot').locator('strong')).toHaveText('photo-3.png');
});

test('internal image file payload never becomes a new upload', async ({page}) => {
  const id = await page.locator('.photo-card').first().getAttribute('data-photo-id');
  const transfer = await page.evaluateHandle(id => {
    const d = new DataTransfer();
    d.setData('application/x-dkp-photo',id!);
    d.items.add(new File(['internal browser image'],'image.png',{type:'image/png'}));
    return d;
  },id);
  let uploads = 0;
  page.on('request',r => { if(r.method()==='POST' && r.url().endsWith('/api/upload')) uploads++; });
  await page.locator('.app').dispatchEvent('drop',{dataTransfer:transfer});
  await page.waitForTimeout(300);
  expect(uploads).toBe(0);
  await expect(page.locator('.photo-card')).toHaveCount(3);
});

test('external files drop into both list layouts and duplicates are skipped', async ({page}) => {
  for (const side of [false,true]) {
    if(side) await page.getByRole('button',{name:'왼쪽 목록',exact:true}).click();
    const name = `photo-${side ? 5 : 4}.png`;
    const bytes = [...fs.readFileSync(path.resolve('e2e/fixtures',name))];
    const transfer = await page.evaluateHandle(({name,bytes}) => {
      const data = new DataTransfer();
      data.items.add(new File([new Uint8Array(bytes)],name,{type:'image/png'}));
      return data;
    },{name,bytes});
    const card = page.locator('.photo-card').first();
    expect(await card.evaluate((el,data) => !el.dispatchEvent(new DragEvent('dragover',{
      dataTransfer:data,bubbles:true,cancelable:true,
    })),transfer)).toBe(true);
    await card.dispatchEvent('drop',{dataTransfer:transfer});
    await expect(page.locator('.photo-card')).toHaveCount(side ? 5 : 4);
    await expect(page.getByRole('button',{name:'+ 사진 추가',exact:true})).toBeEnabled();
    await card.dispatchEvent('drop',{dataTransfer:transfer});
    await expect(page.locator('.status-bar')).toContainText('이미 불러온 사진 1장 건너뜀');
    await expect(page.locator('.photo-card')).toHaveCount(side ? 5 : 4);
  }
  await expect(page.locator('header').getByRole('button',{name:'목록 비우기',exact:true})).toBeVisible();
  await expect(page.locator('footer').getByRole('button',{name:'목록 비우기',exact:true})).toHaveCount(0);
});
