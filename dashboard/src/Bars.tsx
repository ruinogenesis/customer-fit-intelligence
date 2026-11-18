// Each input's share as a bar from a middle line (the profile's split, T404;
// the drivers' comparison, T405): right raises the score, left lowers it.
// Blue is good news and red bad, as the caller says; every value is printed,
// signed. Bars are at most 14px thick with their free end rounded (dataviz
// skill), and scale to the largest share, but never below floor, so small
// shares stay small.
import type { ReactNode } from "react";

import { INPUT_NAMES, INPUTS, type Input } from "./customers";

export function Bars({
  parts,
  label,
  floor,
  format,
  better,
  legend,
}: {
  parts: Record<Input, number>;
  label: string;
  /** The smallest share that fills half the width. */
  floor: number;
  format: (value: number) => string;
  /** Whether a share is good news. */
  better: (value: number) => boolean;
  legend: ReactNode;
}) {
  const scale = Math.max(floor, ...INPUTS.map((input) => Math.abs(parts[input])));
  return (
    <>
      <table className="bars" aria-label={label}>
        <tbody>
          {INPUTS.map((input) => {
            const value = parts[input];
            const width = `${(Math.abs(value) / scale) * 50}%`;
            const side = value > 0 ? "right" : "left";
            return (
              <tr key={input}>
                <th scope="row">{INPUT_NAMES[input]}</th>
                <td className="track">
                  {Math.abs(value) >= 5e-4 * scale && (
                    <span
                      className={`bar ${side} ${better(value) ? "better" : "worse"}`}
                      style={{ width }}
                    />
                  )}
                </td>
                <td className="amount">{format(value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="legend">{legend}</p>
    </>
  );
}
