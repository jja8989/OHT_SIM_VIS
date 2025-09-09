
import React, { useState, useEffect, useRef, forwardRef } from "react";

interface TimeInputProps {
  value: number; // seconds
  onChange: (seconds: number) => void;
  className?: string;
}

const TimeInput = forwardRef<HTMLInputElement, TimeInputProps>(
  ({ value, onChange, className }, ref) => {
    const [str, setStr] = useState(formatTime(value));
    const innerRef = useRef<HTMLInputElement>(null);

    function formatTime(secs: number) {
      const h = String(Math.floor(secs / 3600)).padStart(2, "0");
      const m = String(Math.floor((secs % 3600) / 60)).padStart(2, "0");
      const s = String(secs % 60).padStart(2, "0");
      return `${h}:${m}:${s}`;
    }

    function parseTime(str: string) {
      const [h, m, s] = str.split(":").map((p) => parseInt(p || "0", 10));
      return h * 3600 + m * 60 + s;
    }

    useEffect(() => {
      setStr(formatTime(value));
    }, [value]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
      const inputEl = innerRef.current;
      const pos = inputEl?.selectionStart ?? 0;

      // 숫자 입력
      if (/^[0-9]$/.test(e.key)) {
        e.preventDefault();
        if (pos >= str.length) return;

        let newPos = pos;
        // 콜론 자리는 건너뛰기
        if (str[pos] === ":") newPos++;

        const newStr =
          str.substring(0, newPos) + e.key + str.substring(newPos + 1);

        setStr(newStr);
        onChange(parseTime(newStr));
        moveCaret(newPos + 1);
      }

      // 백스페이스 입력
      if (e.key === "Backspace") {
        e.preventDefault();
        if (pos <= 0) return;

        let newPos = pos;
        if (str[pos - 1] === ":") newPos--;

        const newStr =
          str.substring(0, newPos - 1) + "0" + str.substring(newPos);

        setStr(newStr);
        onChange(parseTime(newStr));
        moveCaret(newPos - 1);
      }
    };

    const moveCaret = (pos: number) => {
      requestAnimationFrame(() => {
        innerRef.current?.setSelectionRange(pos, pos);
      });
    };

    return (
      <input
        ref={(node) => {
          // 외부 ref와 내부 ref 동기화
          if (typeof ref === "function") {
            ref(node);
          } else if (ref) {
            (ref as React.MutableRefObject<HTMLInputElement | null>).current =
              node;
          }
          innerRef.current = node;
        }}
        type="text"
        value={str}
        onKeyDown={handleKeyDown}
        className={`w-32 p-2 rounded-md border text-center font-mono tracking-widest focus:outline-none focus:ring focus:ring-blue-500 ${className}`}
      />
    );
  }
);

TimeInput.displayName = "TimeInput"; // forwardRef 쓰면 이름 지정 필수
export default TimeInput;

