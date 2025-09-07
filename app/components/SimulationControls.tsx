import React from "react";

interface SimulationControlsProps {
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onFaster: () => void;
  onSlower: () => void;
}


const SimulationControls: React.FC<SimulationControlsProps> = ({
  isPlaying,
  onPlay,
  onPause,
  onFaster,
  onSlower,
}) => {

  return (
    <div className="flex flex-col gap-2 p-2">
      <button
        onClick={isPlaying ? onPause : onPlay}
        className={`px-3 py-1 text-white rounded ${
          isPlaying
            ? "bg-yellow-600 hover:bg-yellow-700"
            : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {isPlaying ? "⏸" : "▶"}
      </button>
      <button
        onClick={onFaster}
        className="px-3 py-1 bg-blue-600 text-white rounded"
      >
        ⏩
      </button>
      <button
        onClick={onSlower}
        className="px-3 py-1 bg-blue-600 text-white rounded"
      >
        ⏪
      </button>
    </div>
  );

};

export default SimulationControls;