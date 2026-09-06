import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
const fixture = (i: number) => path.resolve("e2e/fixtures", `photo-${i}.png`);
async function upload(page: Page, n = 8) {
  await page
    .locator("input[type=file]")
    .setInputFiles(Array.from({ length: n }, (_, i) => fixture(i + 1)));
  await expect(page.locator(".photo-card")).toHaveCount(n);
}
async function state(page: Page) {
  return (await page.request.get("/api/state")).json();
}
async function select(page: Page, n: number) {
  await page
    .getByRole("button", { name: `사진 보기 photo-${n}.png`, exact: true })
    .click();
  await expect(page.getByTestId("active-name")).toHaveText(`photo-${n}.png`);
}
async function idle(page: Page) {
  await expect.poll(async () => !(await state(page)).running).toBe(true);
  await expect(
    page.getByRole("button", { name: "현재 정합", exact: true }),
  ).toBeEnabled();
}
test.beforeEach(async ({ page }) => {
  await page.request.post("/api/reset");
  await page.goto("/");
});

test("click first, then A selects anchor or Z commits mask without choosing a tool", async ({ page }) => {
  await upload(page, 2);
  await expect(page.locator('input[type=file]')).toHaveCount(1);
  const panes = page.locator('.photo-viewport');
  await panes.first().click({ position: { x: 160, y: 140 } });
  await expect(page.locator('[data-anchor="candidate"]')).toBeVisible();
  await page.keyboard.press('a');
  await panes.last().click({ position: { x: 160, y: 140 } });
  await page.keyboard.press('a');
  await expect.poll(async () => {
    const s = await state(page);
    return (await (await page.request.get(`/api/anchors/${s.images[1].id}`)).json()).pairs.length;
  }).toBe(1);
  let s = await state(page);
  expect(s.images.every((p: any) => !p.mask_ready)).toBe(true);
  await panes.last().click({ position: { x: 200, y: 160 } });
  await expect(page.getByRole('button', {name: '개체 확정 Z', exact:true})).toBeEnabled();
  await page.keyboard.press('z');
  await expect.poll(async () => (await state(page)).images[1].n_objects).toBe(1);
  await page.keyboard.press('Control+z');
  await expect.poll(async () => (await state(page)).images[1].n_objects).toBe(0);
});

test("name order is independent of default fixed; drag replaces the top fixed slot", async ({page}) => {
  const buffer = fs.readFileSync(fixture(1));
  await page.locator('input[type=file]').setInputFiles(['z.png','photo-10.png','photo-2.png'].map(name => ({name, mimeType:'image/png', buffer})));
  await expect(page.locator('.photo-card')).toHaveCount(3);
  await expect(page.getByTestId('fixed-slot')).toContainText('z.png');
  await expect(page.locator('.card-copy strong')).toHaveText(['photo-2.png','photo-10.png','z.png']);
  await page.locator('.photo-card').filter({has:page.getByRole('button',{name:'사진 보기 photo-10.png',exact:true})}).dragTo(page.getByTestId('fixed-slot'));
  await expect(page.getByTestId('fixed-slot')).toContainText('photo-10.png');
  await expect(page.locator('.photo-card')).toHaveCount(3);
  await page.getByRole('button',{name:'↶ 되돌리기',exact:true}).click();
  await expect(page.getByTestId('fixed-slot')).toContainText('z.png');
});

test("fixed replacement leaves previous result and reference visible until registering again", async ({page}) => {
  await upload(page,3);
  await select(page,2);
  await page.getByRole('button',{name:'현재 정합',exact:true}).click();
  await idle(page);
  const old = (await state(page)).images[1].result.id;
  await page.locator('.photo-card').filter({has:page.getByRole('button',{name:'사진 보기 photo-3.png',exact:true})}).dragTo(page.getByTestId('fixed-slot'));
  await expect(page.getByTestId('fixed-slot')).toContainText('photo-3.png');
  await expect(page.getByTestId('active-name')).toHaveText('photo-2.png');
  await expect(page.locator('.context-toolbar')).toContainText(['결과 기준: photo-1.png']);
  expect((await state(page)).images[1].result.id).toBe(old);
  await page.getByRole('button',{name:'현재 정합',exact:true}).click();
  await idle(page);
  await expect(page.locator('.context-toolbar')).toContainText(['결과 기준: photo-3.png']);
  expect((await state(page)).images[1].result.id).not.toBe(old);
});

test("upload, fifth result, preserve checks and navigation, save selected", async ({
  page,
}) => {
  await upload(page);
  let s = await state(page);
  expect(s.fixed_id).toBe(s.images[0].id);
  await select(page, 2);
  await page.getByRole("button", { name: "현재 정합", exact: true }).click();
  await idle(page);
  await page
    .getByRole("checkbox", { name: "일괄 선택 photo-5.png", exact: true })
    .check();
  await page
    .getByRole("button", { name: "선택 1장 정합", exact: true })
    .click();
  await idle(page);
  await expect(page.getByTestId("active-name")).toHaveText("photo-5.png");
  await expect(
    page.getByRole("checkbox", { name: "일괄 선택 photo-5.png", exact: true }),
  ).toBeChecked();
  await page
    .getByRole("button", { name: "선택 1장 정합", exact: true })
    .click();
  await select(page, 8);
  await idle(page);
  await expect(page.getByTestId("active-name")).toHaveText("photo-8.png");
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "dkp-e2e-save-"));
  await page.route("**/api/select_folder", (route) =>
    route.fulfill({ json: { path: out } }),
  );
  await page
    .getByRole("button", { name: "선택 결과 저장 (1)", exact: true })
    .click();
  await expect.poll(() => fs.readdirSync(out).length).toBe(1);
  expect(fs.readdirSync(out)[0]).toContain("photo-5");
});
test("mask Z/X undo restores prompts; C and form shortcuts do nothing", async ({
  page,
}) => {
  await upload(page, 3);
  await page.getByRole("button", { name: "점·마스크", exact: true }).click();
  const pane = page.locator(".photo-viewport").nth(1);
  await pane.click({ position: { x: 150, y: 150 } });
  await expect(page.locator('[data-anchor="candidate"]')).toBeVisible();
  await expect(page.getByRole("button", { name: /개체 확정/ })).toBeEnabled();
  await pane.press("KeyZ");
  await expect
    .poll(async () => (await state(page)).images[1].n_objects)
    .toBe(1);
  await expect(
    page.getByRole("button", { name: /마스크 초기화/ }),
  ).toBeEnabled();
  await pane.press("KeyX");
  await expect
    .poll(async () => (await state(page)).images[1].n_objects)
    .toBe(0);
  await pane.press("Control+z");
  await expect
    .poll(async () => (await state(page)).images[1].n_objects)
    .toBe(1);
  await pane.press("Control+z");
  await expect
    .poll(async () => (await state(page)).images[1].mask_points.length)
    .toBe(0);
  let rev = (await state(page)).revision;
  await pane.press("KeyC");
  expect((await state(page)).revision).toBe(rev);
  await page.getByRole("searchbox", { name: "파일명 검색" }).fill("aXZ");
  expect((await state(page)).revision).toBe(rev);
});
test("anchors and changing fixed preserve pair state; review queue advances", async ({
  page,
}) => {
  await upload(page, 4);
  await page.locator(".work-surface").focus();
  await page.keyboard.press("KeyA");
  await page
    .locator(".photo-viewport")
    .nth(0)
    .click({ position: { x: 150, y: 150 } });
  await page.keyboard.press("KeyA");
  await page
    .locator(".photo-viewport")
    .nth(1)
    .click({ position: { x: 180, y: 150 } });
  await page.keyboard.press("KeyA");
  await expect(page.locator(".anchor-dot")).toHaveCount(2);
  await page.locator(".work-surface").focus();
  await page.keyboard.press("KeyD");
  await expect(page.locator(".anchor-dot")).toHaveCount(0);
  await page.keyboard.press("Control+z");
  await expect(page.locator(".anchor-dot")).toHaveCount(2);
  await select(page, 3);
  await page
    .getByRole("button", { name: "이 사진을 기준으로 지정", exact: true })
    .click();
  await expect(page.locator(".reference-summary")).toContainText("photo-3.png");
  await page.getByRole("button", { name: "↶ 되돌리기", exact: true }).click();
  await expect(page.locator(".reference-summary")).toContainText("photo-1.png");
  await select(page, 2);
  await expect(page.locator(".anchor-dot")).toHaveCount(2);
  await page.getByRole("button", { name: "전체 정합", exact: true }).click();
  await idle(page);
  await page
    .getByRole("combobox", { name: "사진 필터" })
    .selectOption("review");
  await select(page, 2);
  await page
    .getByRole("button", { name: "확인하고 다음 →", exact: true })
    .click();
  await expect(page.getByTestId("active-name")).toHaveText("photo-3.png");
  await page
    .getByRole("button", { name: "확인하고 다음 →", exact: true })
    .click();
  await expect(page.getByTestId("active-name")).toHaveText("photo-4.png");
});
test("original ROI, zoom retention, responsive workspace and editor apply", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await upload(page, 3);
  const requests: string[] = [];
  page.on("request", (req) => {
    if (req.url().includes("/region?")) requests.push(req.url());
  });
  await page.getByRole("button", { name: "100%", exact: true }).click();
  await expect.poll(() => requests.length).toBeGreaterThan(0);
  await page.getByRole("button", { name: "부분 확대", exact: true }).click();
  await expect(page.locator(".loupe")).toHaveCount(2);
  for (const [w, h] of [
    [1366, 768],
    [1920, 1080],
    [1093, 614],
    [911, 512],
  ]) {
    await page.setViewportSize({ width: w, height: h });
    if (w < 950 && (await page.locator(".inspector").isVisible()))
      await page.getByRole("button", { name: "정보 패널 닫기" }).click();
    for (const name of ["현재 정합", "전체 정합"]) {
      const box = await page
        .getByRole("button", { name, exact: true })
        .boundingBox();
      expect(box).not.toBeNull();
      expect(box!.y + box!.height).toBeLessThanOrEqual(h);
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({ path: `e2e/artifacts/workspace-${w}x${h}.png` });
  }
  await page.setViewportSize({ width: 1366, height: 768 });
  await page.getByRole("button", { name: "기준 편집", exact: true }).click();
  await expect(page.locator("[data-tool=fixed-editor]")).toBeVisible();
  await page.getByRole("button", { name: "↷ 90°", exact: true }).click();
  await page.getByRole("button", { name: "기준에 적용", exact: true }).click();
  await expect(page.locator("[data-tool=fixed-editor]")).toHaveCount(0);
  let s = await state(page);
  expect(s.images[0].full_w).toBe(1000);
  expect(s.images[0].full_h).toBe(1600);
  await page.getByRole("button", { name: "↶ 되돌리기", exact: true }).click();
  await expect
    .poll(async () => (await state(page)).images[0].full_w)
    .toBe(1600);
  expect(errors).toEqual([]);
});

test("old result uses its pinned dimensions and high-resolution overlay; manual draft persists", async ({
  page,
}) => {
  await upload(page, 3);
  await page.getByRole("button", { name: "현재 정합", exact: true }).click();
  await idle(page);
  await page.getByRole("button", { name: "기준 편집", exact: true }).click();
  await page.getByRole("button", { name: "↷ 90°", exact: true }).click();
  await page.getByRole("button", { name: "기준에 적용", exact: true }).click();
  await expect(page.locator("[data-tool=fixed-editor]")).toHaveCount(0);
  await page.getByRole("button", { name: "현재 정합", exact: true }).click();
  await idle(page);
  await page
    .getByRole("button", { name: "이전 정합 결과 비교", exact: true })
    .click();
  await expect(page.locator(".photo-plane").first()).toHaveCSS(
    "width",
    "1600px",
  );
  await page.getByRole("button", { name: "와이프", exact: true }).click();
  await page.getByRole("button", { name: "100%", exact: true }).click();
  await expect.poll(() => page.locator("img.roi-layer").count()).toBe(2);
  await page.getByRole("button", { name: "미세조정", exact: true }).click();
  await expect(page.locator(".photo-plane").first()).toHaveCSS(
    "width",
    "1000px",
  );
  const x = page.locator(".adjustment input[type=number]").first();
  await x.fill("12");
  await x.press("Tab");
  await select(page, 3);
  await select(page, 2);
  await expect(
    page.locator(".adjustment input[type=number]").first(),
  ).toHaveValue("12");
  await page.getByRole("button", { name: "조정 적용", exact: true }).click();
  await expect
    .poll(async () => (await state(page)).images[1].result.manual_adjusted)
    .toBe(true);
});

test("review response does not override navigation; stop finishes current item", async ({
  page,
}) => {
  await upload(page, 4);
  await page.getByRole("button", { name: "전체 정합", exact: true }).click();
  await idle(page);
  await select(page, 2);
  let unblock!: () => void;
  const barrier = new Promise<void>((r) => (unblock = r));
  await page.route("**/api/result/*/review", async (route) => {
    await barrier;
    await route.continue();
  });
  await page
    .getByRole("button", { name: "확인하고 다음 →", exact: true })
    .click();
  await select(page, 4);
  unblock();
  await expect
    .poll(async () => (await state(page)).images[1].result.review_status)
    .toBe("confirmed");
  await expect(page.getByTestId("active-name")).toHaveText("photo-4.png");
  await page.getByRole("button", { name: "전체 정합", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "남은 작업 중지", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "남은 작업 중지", exact: true })
    .click();
  await idle(page);
  const s = await state(page);
  expect(s.job.cancelled).toBe(true);
  expect(Object.values(s.job.items)).toContain("cancelled");
});

test("pending adjustment keeps another photo draft; export stays bound to job reference", async ({
  page,
}) => {
  await upload(page, 3);
  await page.getByRole("button", { name: "전체 정합", exact: true }).click();
  await idle(page);
  await select(page, 3);
  await page.getByRole("button", { name: "미세조정", exact: true }).click();
  await page.locator(".adjustment input[type=number]").first().fill("27");
  await page.locator(".adjustment input[type=number]").first().press("Tab");
  await select(page, 2);
  await page.locator(".adjustment input[type=number]").first().fill("12");
  await page.locator(".adjustment input[type=number]").first().press("Tab");
  let unblock!: () => void;
  const barrier = new Promise<void>((r) => (unblock = r));
  await page.route("**/api/result/*/adjust", async (route) => {
    await barrier;
    await route.continue();
  });
  await page.getByRole("button", { name: "조정 적용", exact: true }).click();
  await select(page, 3);
  unblock();
  await expect
    .poll(async () => (await state(page)).images[1].result.manual_adjusted)
    .toBe(true);
  await expect(
    page.locator(".adjustment input[type=number]").first(),
  ).toHaveValue("27");
  await page
    .getByRole("button", { name: "이 사진을 기준으로 지정", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "이번 작업 결과 저장", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "기준 편집", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "목록 비우기", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", {
      name: "목록에서 제거 photo-3.png",
      exact: true,
    }),
  ).toBeDisabled();
});

test("direct adjustment is uniform and one drag is one undo step", async ({ page }) => {
  await upload(page, 2);
  await select(page, 2);
  await page.getByRole("button", { name: "현재 정합", exact: true }).click();
  await idle(page);
  await page.getByRole("button", { name: "미세조정", exact: true }).click();
  const frame = page.locator(".adjust-frame");
  const box = (await frame.boundingBox())!;
  const x = page.locator(".adjustment input[type=number]").first();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2 + 10, { steps: 8 });
  await page.mouse.up();
  await expect(x).not.toHaveValue("0");
  await page.keyboard.press("Control+z");
  await expect(x).toHaveValue("0");
  await page.keyboard.down("Space");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 20, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  await page.keyboard.up("Space");
  await expect(x).toHaveValue("0");
  const corner = frame.locator('[data-adjust="scale"]').first();
  const h = (await corner.boundingBox())!;
  await page.mouse.move(h.x + h.width / 2, h.y + h.height / 2);
  await page.mouse.down();
  await page.mouse.move(h.x + h.width / 2 + 20, h.y + h.height / 2 + 20, { steps: 5 });
  await page.mouse.up();
  const scale = page.locator(".adjustment input[type=number]").nth(3);
  await expect(scale).not.toHaveValue("1");
  await page.keyboard.press("Control+z");
  await expect(scale).toHaveValue("1");
});

test("thirty-photo workspace reports navigation latency and browser heap", async ({ page }) => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable");
  const buffer = fs.readFileSync(fixture(1));
  const start = Date.now();
  await page.locator("input[type=file]").setInputFiles(Array.from({ length: 30 }, (_, i) => ({
    name: `photo-${i + 1}.png`, mimeType: "image/png", buffer,
  })));
  await expect(page.locator(".photo-card")).toHaveCount(30);
  const uploadMs = Date.now() - start;
  const latencies: number[] = [];
  for (const n of [30, 15, 2, 29, 10]) {
    const t = Date.now();
    await select(page, n);
    latencies.push(Date.now() - t);
  }
  const metrics = await cdp.send("Performance.getMetrics");
  const heap = metrics.metrics.find((m: { name: string }) => m.name === "JSHeapUsedSize")!.value;
  const report = { photos: 30, imageSize: "1600x1000 synthetic PNG", uploadMs, navigationMs: latencies,
    jsHeapMiB: Math.round(heap / 1048576 * 10) / 10,
    limitation: "Chromium JS heap only; excludes decoded image/GPU/Python native memory. No clinical registration benchmark." };
  fs.mkdirSync("e2e/artifacts", { recursive: true });
  fs.writeFileSync("e2e/artifacts/performance.json", JSON.stringify(report, null, 2));
  console.log("WORKSPACE PERFORMANCE", JSON.stringify(report));
  await expect(page.getByTestId("active-name")).toHaveText("photo-10.png");
});
