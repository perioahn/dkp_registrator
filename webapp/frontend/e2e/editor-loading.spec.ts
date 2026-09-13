import {test,expect} from '@playwright/test'
import path from 'node:path'

test('fixed editor reuses its prefetched preview and applies at source resolution',async({page})=>{
  await page.request.post('/api/reset')
  await page.goto('/')
  // A larger source makes accidental preview-sized export observable.
  const bytes=await page.evaluate(()=>{
    const c=document.createElement('canvas');c.width=3600;c.height=2400
    const g=c.getContext('2d')!;g.fillStyle='#a7826f';g.fillRect(0,0,c.width,c.height)
    g.fillStyle='#eee';g.fillRect(200,200,300,300)
    return c.toDataURL('image/jpeg',.92).split(',')[1]
  })
  let previews=0
  await page.route(/\/api\/image\/[^/]+\/source-preview$/,async route=>{previews++;await route.continue()})
  await page.locator('input[type=file]').setInputFiles({name:'large.jpg',mimeType:'image/jpeg',buffer:Buffer.from(bytes,'base64')})
  await expect(page.locator('.photo-card')).toHaveCount(1)
  await expect.poll(()=>previews).toBe(1)
  let release!:()=>void
  const hold=new Promise<void>(r=>release=r)
  let originals=0
  await page.route(/\/api\/image\/[^/]+\/source$/,async route=>{originals++;await hold;await route.continue()})
  try {
    const start=performance.now()
    await page.getByRole('button',{name:'기준 편집',exact:true}).click()
    const editor=page.locator('[data-tool=fixed-editor]')
    await expect(editor.locator('canvas')).toHaveAttribute('width','1600')
    await expect(editor).toHaveAttribute('aria-busy','false')
    await expect(page.locator('.inspector')).toBeHidden()
    await expect(page.locator('.photo-browser')).toBeHidden()
    const openMs=performance.now()-start
    expect(originals).toBe(0)
    await expect(editor.getByRole('heading',{name:'빛과 톤'})).toBeVisible()
    await editor.getByRole('button',{name:'편집 취소',exact:true}).click()
    await expect(editor).toHaveCount(0)
    await page.getByRole('button',{name:'기준 편집',exact:true}).click()
    await expect(editor).toHaveAttribute('aria-busy','false')
    expect(previews).toBe(1)
    await editor.getByRole('slider',{name:'밝기',exact:true}).fill('15')
    await editor.getByRole('button',{name:'↷ 90°',exact:true}).click()
    await page.screenshot({path:'test-results/integrated-editor.png'})
    console.log('EDITOR OPEN WITHOUT ORIGINAL',JSON.stringify({milliseconds:openMs,sourceWidth:3600,previewWidth:1600}))
    await editor.getByRole('button',{name:'기준에 적용',exact:true}).click()
    await expect.poll(()=>originals).toBe(1)
    await expect(editor.getByRole('button',{name:'편집 취소',exact:true})).toBeDisabled()
  } finally {release()}
  await expect(page.locator('[data-tool=fixed-editor]')).toHaveCount(0)
  await expect(page.locator('.photo-browser')).toBeVisible()
  const state=await (await page.request.get('/api/state')).json()
  expect([state.images[0].full_w,state.images[0].full_h]).toEqual([2400,3600])
  expect(state.images[0].edits.brightness).toBe(15)
  await page.getByRole('button',{name:'↶ 되돌리기',exact:true}).click()
  await expect.poll(async()=>{
    const s=await (await page.request.get('/api/state')).json()
    return [s.images[0].full_w,s.images[0].full_h]
  }).toEqual([3600,2400])
})

test('crop from a scaled source preview remains in original pixel coordinates',async({page})=>{
  await page.request.post('/api/reset');await page.goto('/')
  await page.locator('input[type=file]').setInputFiles(path.resolve('e2e/fixtures/photo-1.png'))
  await expect(page.locator('.photo-card')).toHaveCount(1)
  await page.getByRole('button',{name:'기준 편집',exact:true}).click()
  const editor=page.locator('[data-tool=fixed-editor]')
  await expect(editor).toHaveAttribute('aria-busy','false')
  await editor.getByRole('button',{name:'✂ 크롭',exact:true}).click()
  await editor.getByRole('button',{name:'1:1',exact:true}).click()
  await editor.getByRole('button',{name:'✓ 적용',exact:true}).click()
  await editor.getByRole('button',{name:'기준에 적용',exact:true}).click()
  await expect(editor).toHaveCount(0)
  const state=await (await page.request.get('/api/state')).json()
  expect(state.images[0].full_w).toBeGreaterThan(500)
  expect(state.images[0].full_w).toBe(state.images[0].full_h)
})
