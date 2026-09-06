import { describe, expect, it } from 'vitest'
import { editGeometry, freshEdits } from './photoEdits'

describe('source pixel centers to full-resolution edited geometry', () => {
  it('maps the exact source pixel under ninety-degree rotation and reflection', () => {
    const e = { ...freshEdits(), rot90: 1 }
    const { G, width, height } = editGeometry(7, 5, e)
    expect([width,height]).toEqual([5,7])
    const map = (x:number,y:number) => [G[0][0]*x+G[0][1]*y+G[0][2], G[1][0]*x+G[1][1]*y+G[1][2]]
    expect(map(0,0)[0]).toBeCloseTo(4)
    expect(map(0,0)[1]).toBeCloseTo(0)
    expect(map(6,4)[0]).toBeCloseTo(0)
    expect(map(6,4)[1]).toBeCloseTo(6)
    expect(editGeometry(7,5,{ ...freshEdits(),flipH:true }).G[0]).toEqual([-1,0,6])
  })
  it('never shears or scales individual axes across rotations, flips, fine angles and crops', () => {
    for (let rot90=0;rot90<4;rot90++) for (const flipH of [false,true]) for (const flipV of [false,true]) for (const fineDeg of [-15,-7.3,0,9.1,15]) {
      const e={...freshEdits(),rot90,flipH,flipV,fineDeg,crop:{x:11.2,y:7.8,w:170.7,h:93.2}}
      const { G,width,height }=editGeometry(643,479,e)
      const [a,b]=G[0], [c,d]=G[1]
      expect(a*a+c*c).toBeCloseTo(1,12); expect(b*b+d*d).toBeCloseTo(1,12)
      expect(a*b+c*d).toBeCloseTo(0,12)
      expect([width,height]).toEqual([171,93])
    }
  })
  it('tone changes leave G and output size unchanged', () => {
    expect(editGeometry(1023,767,{...freshEdits(),brightness:90,contrast:-50})).toEqual(editGeometry(1023,767,freshEdits()))
  })
})
