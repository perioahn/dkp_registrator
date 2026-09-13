import {test, expect} from '@playwright/test';

test('GPU download, extraction, destination, completion and requirements', async ({page}) => {
  const gpu:any = {platform:'win32', device:'cpu', frozen:true, gpu_name:'NVIDIA test', installing:true, phase:'download', pkg:'torch', done:1048576, total:2097152, install_dir:'C:\\Users\\User\\AppData\\Local\\DKPRegistrator\\cuda', can_restart:true};
  await page.route('**/api/gpu', route=>route.fulfill({json:gpu}));
  await page.goto('/');
  await page.getByText('정합 설정', {exact:true}).click();
  await expect(page.locator('.gpu-install')).toContainText('50% · 1.0 / 2.0 MB');
  await expect(page.locator('.gpu-install')).toContainText(gpu.install_dir);
  await page.getByText('GPU 사용 조건 ⓘ').click();
  await expect(page.locator('.gpu-requirements')).toContainText('CUDA Toolkit은 별도로 설치하지 않아도');
  gpu.phase='extract'; gpu.pkg='torchvision'; gpu.total=0;
  await page.reload();
  await page.getByText('정합 설정', {exact:true}).click();
  await expect(page.locator('.gpu-install')).toContainText('압축 해제 · torchvision (2/2)');
  await expect(page.locator('progress')).not.toHaveAttribute('value');
  gpu.phase='done'; gpu.installing=false; gpu.installed=true;
  await page.reload();
  await page.getByText('정합 설정', {exact:true}).click();
  await expect(page.getByRole('button',{name:'GPU 적용하고 다시 시작'})).toBeEnabled();
  await expect(page.locator('.gpu-install')).toContainText('설치 완료');
});
