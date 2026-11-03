import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Info, LevelBadge } from "./components/ui";

describe("status badge", () => {
  it("always pairs colour with an icon and a text label", () => {
    render(<><LevelBadge level="alert" /><LevelBadge level="unusual" /><LevelBadge level="normal" /></>);
    expect(screen.getByText("Model alert")).toBeInTheDocument();
    expect(screen.getByText("Unusual activity")).toBeInTheDocument();
    expect(screen.getByText("Normal conditions")).toBeInTheDocument();
  });
});

describe("tooltip", () => {
  it("shows its plain-English explanation on hover and keyboard focus", () => {
    render(<Info text="VIX: the market's expectation of volatility." />);
    expect(screen.queryByRole("tooltip")).toBeNull();
    fireEvent.focus(screen.getByRole("button", { name: "What is this?" }));
    expect(screen.getByRole("tooltip")).toHaveTextContent("expectation of volatility");
  });
});
