import { describe, it, expect } from "vitest";
import {
  preferredResult,
  completionSelection,
  shortcut,
  mapPoint,
  inversePoint,
  nextReview,
  zoomAt,
} from "./workspace";

describe("registration selection snapshots", () => {
  it("opens only the requested fifth photo, never an old first result", () => {
    expect(preferredResult(["5"], "1")).toBe("5");
    expect(
      completionSelection(
        { id: "job", preferred: "5", navigation: 4 },
        "job",
        4,
        "1",
      ),
    ).toBe("5");
  });
  it("keeps navigation and ignores another tab or stale job", () => {
    expect(
      completionSelection(
        { id: "job", preferred: "5", navigation: 4 },
        "job",
        5,
        "8",
      ),
    ).toBe("8");
    expect(
      completionSelection(
        { id: "new", preferred: "5", navigation: 4 },
        "old",
        4,
        "8",
      ),
    ).toBe("8");
    expect(completionSelection(null, "job", 4, "8")).toBe("8");
  });
  it("walks a snapshot review queue even when reviewed rows leave a filter", () => {
    expect(nextReview(["2", "5", "8"], "5", new Set(["2", "5", "8"]))).toBe(
      "8",
    );
    expect(nextReview(["2", "5", "8"], "2", new Set(["2", "8"]))).toBe("8");
  });
});
describe("focus-scoped shortcuts", () => {
  const key = (code: string, extra: Record<string, unknown> = {}) => ({
    code,
    ctrlKey: false,
    metaKey: false,
    altKey: false,
    shiftKey: false,
    isComposing: false,
    repeat: false,
    ...extra,
  });
  it("modifiers precede letter commands, C has no action", () => {
    expect(shortcut(key("KeyZ", { ctrlKey: true }), "mask", false)).toBe(
      "undo",
    );
    expect(
      shortcut(key("KeyZ", { metaKey: true, shiftKey: true }), "mask", false),
    ).toBe("redo");
    expect(shortcut(key("KeyZ"), "mask", false)).toBe("confirm");
    expect(shortcut(key("KeyC"), "mask", false)).toBe(null);
    expect(shortcut(key("KeyX", { ctrlKey: true }), "mask", false)).toBe(null);
  });
  it("input, composition, held keys and wrong tool never mutate", () => {
    expect(shortcut(key("KeyX"), "mask", true)).toBe(null);
    expect(shortcut(key("KeyZ", { isComposing: true }), "mask", false)).toBe(
      null,
    );
    expect(shortcut(key("KeyZ", { repeat: true }), "mask", false)).toBe(null);
    expect(shortcut(key("KeyZ"), "compare", false)).toBe("confirm");
    expect(shortcut(key("KeyZ"), "adjust", false)).toBe(null);
  });
});
describe("coordinates", () => {
  it("round trips rotation flip and crop without cumulative coordinate errors", () => {
    const G = [
        [0, -1, 790],
        [-1, 0, 1190],
        [0, 0, 1],
      ],
      p: [number, number] = [202, 404];
    expect(mapPoint(G, p)).toEqual([386, 988]);
    expect(inversePoint(G, mapPoint(G, p))).toEqual(p);
  });
  it("wheel zoom preserves point beneath pointer", () => {
    const v = { zoom: 1, cx: 0.5, cy: 0.5 };
    const n = zoomAt(v, 2, 700, 250, 1000, 500, 1000, 500, 1);
    expect((700 - 500) / 2 + n.cx * 1000).toBeCloseTo(700);
    expect((250 - 250) / 2 + n.cy * 500).toBeCloseTo(250);
  });
});
