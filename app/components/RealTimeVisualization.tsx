import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import * as d3 from 'd3';

interface DataPoint {
  positionX: number;
  positionY: number;
  port: string;
}

const RealTimeVisualization: React.FC = () => {
  const [data, setData] = useState<DataPoint[]>([]);
  const socketRef = useRef<Socket | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    socketRef.current = io('http://localhost:5000');

    socketRef.current.on('simulation_update', (newData: DataPoint) => {
      setData(prevData => [...prevData, newData]);
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  useEffect(() => {
    const svg = d3.select(svgRef.current)
      .attr('width', 800)
      .attr('height', 600)
      .style('border', '1px solid black');

    svg.selectAll('*').remove();

    svg.selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => d.positionX * 10)
      .attr('cy', d => d.positionY * 10)
      .attr('r', 5)
      .attr('fill', 'red');

    svg.selectAll('text')
      .data(data)
      .enter()
      .append('text')
      .attr('x', d => d.positionX * 10)
      .attr('y', d => d.positionY * 10 - 10)
      .text(d => d.port)
      .attr('font-size', '12px')
      .attr('fill', 'black');
  }, [data]);

  return (
    <div>
      <h2>Real-Time Simulation</h2>
      <svg ref={svgRef}></svg>
    </div>
  );
};

export default RealTimeVisualization;
