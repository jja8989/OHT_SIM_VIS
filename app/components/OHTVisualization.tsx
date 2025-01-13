import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';
import pako from 'pako'; // Gzip 압축 해제를 위해 pako 라이브러리 사용


const socket = io('http://localhost:5001');

interface Node {
    id: string;
    x: number;
    y: number;
}

interface Rail {
    from: string;
    to: string;
    count: number;  // Add count to the Rail interface
    avg_speed : number;
}

interface Port {
    name: string;
    x: number;
    y: number;
    rail_line: string;
}

interface OHT {
    id: string;
    x: number;
    y: number;
    time: number;  // Include time in OHT interface
    source: string;
    dest: string;
}

interface LayoutData {
    nodes: Node[];
    rails: Rail[];
    ports: Port[];
}

interface OHTVisualizationProps {
    data: LayoutData;
}

        // Color scale for rail counts
const colorScale = d3.scaleLinear()
            .domain([0, 1])
            .range([
                '#0000ff',
                '#ff0000'
                ]);

const decompressData = (compressedData: string) => {
    try {
        const decodedData = atob(compressedData); // Base64 디코딩
        const byteArray = new Uint8Array(decodedData.split('').map(char => char.charCodeAt(0)));
        const jsonData = pako.inflate(byteArray, { to: 'string' }); // Gzip 해제
        return JSON.parse(jsonData); // JSON 파싱
    } catch (error) {
        console.error('Error decompressing data:', error);
        return null;
    }
};
                

const OHTVisualization: React.FC<OHTVisualizationProps> = ({ data }) => {
    const [maxTime, setMaxTime] = useState(4000);
    const [isRunning, setIsRunning] = useState(false);
    const svgRef = useRef<SVGSVGElement | null>(null);
    const gRef = useRef<SVGGElement | null>(null);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const zoomTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity.translate(100, 50).scale(0.5));
    const ohtQueues = useRef<Map<string, OHT[]>>(new Map()); // Use ohtQueues to track last positions
    const processingOHTQueues = useRef<Map<string, boolean>>(new Map());
    const railsRef = useRef<Rail[]>(data.rails); // Maintain a reference to the rails
    const [selectedRail, setSelectedRail] = useState<{ rail: Rail; x: number; y: number } | null>(null);
    const [displayMode, setDisplayMode] = useState<'count' | 'avg_speed'>('count'); // 기본은 count
    const edgeQueues = useRef<Map<string, Rail[]>>(new Map()); // Rails 큐 생성
    const processingEdgeQueues = useRef<Map<string, boolean>>(new Map()); // Rails 처리 상태




    useEffect(() => {
        const svg = d3.select(svgRef.current)
            .attr('width', '100%')  // Set the width of the SVG to be responsive
            .attr('height', '100%'); // Set the height of the SVG to be responsive

        const margin = { top: 50, right: 50, bottom: 50, left: 50 };

        const g = d3.select(gRef.current);

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.5, 5])  // Define the scale extent for zooming
            .translateExtent([[-100, -100], [2400, 1300]])  // Define the panning extent
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
                zoomTransformRef.current = event.transform;  // Track the current zoom transform
            });

        svg.call(zoom).call(zoom.transform, zoomTransformRef.current);  // Apply the initial zoom transform
        zoomRef.current = zoom;

        const { nodes, rails, ports } = data;

        // Initialize rail counts
        rails.forEach(rail => {
            rail.count = 0;  // Initialize count for each rail
        });

        // Find max values for scaling
        const maxX = d3.max(nodes, d => d.x) || 1;
        const maxY = d3.max(nodes, d => d.y) || 1;

        // Create scales
        const yScale = d3.scaleLinear().domain([0, maxY]).range([0, 1200 - margin.top - margin.bottom]);

        // Scale function for nodes and ports
        const scalePosition = (d: { x: number; y: number }) => ({
            x: yScale(d.x),
            y: yScale(d.y)
        });

        const objectToString = (obj: any) => {
            return Object.entries(obj).map(([key, value]) => `${key}: ${value}`).join('\n');
        };

        const tooltip = d3.select('#tooltip');

        const showTooltip = (event: MouseEvent, content: string) => {
            tooltip.style('visibility', 'visible')
                .style('left', `${event.pageX + 5}px`)
                .style('top', `${event.pageY + 5}px`)
                .html(content.replace(/\n/g, '<br>'));
        };

        const hideTooltip = () => {
            tooltip.style('visibility', 'hidden');
        };



        // Draw rails
        const railLines = g.selectAll('.rail')
            .data(rails)
            .enter()
            .append('line')
            .attr('class', 'rail')
            .attr('x1', d => scalePosition(nodes.find(n => n.id === d.from)!).x)
            .attr('y1', d => scalePosition(nodes.find(n => n.id === d.from)!).y)
            .attr('x2', d => scalePosition(nodes.find(n => n.id === d.to)!).x)
            .attr('y2', d => scalePosition(nodes.find(n => n.id === d.to)!).y)
            .attr('stroke-width', 2.5)
            .attr('stroke', d => colorScale(d.count)) // Set initial stroke color
            .on('mouseover', (event, d) => showTooltip(event, objectToString(d)))
            .on('mouseout', hideTooltip)
            .on('click', (event, d) => {
                event.stopPropagation();
                const fromNode = nodes.find(n => n.id === d.from);
                const toNode = nodes.find(n => n.id === d.to);
                if (fromNode && toNode) {
                    const midpoint = {
                        x: (scalePosition(fromNode).x + scalePosition(toNode).x) / 2,
                        y: (scalePosition(fromNode).y + scalePosition(toNode).y) / 2,
                    };
                    setSelectedRail({ rail: d, ...midpoint });
                }
            });

        // Draw nodes
        g.selectAll('.node')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('cx', d => scalePosition(d).x)
            .attr('cy', d => scalePosition(d).y)
            .attr('r', 2)
            .attr('fill', 'red')
            .on('mouseover', (event, d) => showTooltip(event, objectToString(d)))
            .on('mouseout', hideTooltip);

        // Draw ports
        g.selectAll('.port')
            .data(ports)
            .enter()
            .append('circle')
            .attr('class', 'port')
            .attr('cx', d => scalePosition(d).x)
            .attr('cy', d => scalePosition(d).y)
            .attr('r', 1.5)
            .attr('fill', 'green')
            .on('mouseover', (event, d) => showTooltip(event, objectToString(d)))
            .on('mouseout', hideTooltip);

        const updateRailColor = (rail: Rail) => {
            const value = displayMode === 'count' 
                ? rail.count / 100 // `count` 정규화 (최대 100 기준)
                : rail.avg_speed / 1500; // `avg_speed` 정규화 (최대 1500 기준)
    
            d3.selectAll('.rail')
                .filter((d: Rail) => d.from === rail.from && d.to === rail.to)
                .transition()
                .duration(500) // 부드럽게 색상 변경
                .attr('stroke', colorScale(value));
        };

        const handleOHTUpdate = (data: { data: string }) => {
            const decompressedData = decompressData(data.data);
            if (!decompressedData) return;

            const { time, oht_positions, edges } = decompressedData;

            if (time == maxTime){
                setIsRunning(false);
            }
        
            oht_positions.forEach((updatedOHT) => {
                if (!ohtQueues.current.has(updatedOHT.id)) {
                    ohtQueues.current.set(updatedOHT.id, []);
                }
                const queue = ohtQueues.current.get(updatedOHT.id)!;
                queue.push({ ...updatedOHT, time });
            });
        
            // Rails 업데이트
            edges.forEach((edge) => {
                const edgeKey = `${edge.from}-${edge.to}`;
                if (!edgeQueues.current.has(edgeKey)) {
                    edgeQueues.current.set(edgeKey, []);
                }
                const queue = edgeQueues.current.get(edgeKey)!;
                queue.push({ ...edge, time });
            });
        };


                 
        const processOHTQueue = (ohtId: string) => {
            const queue = ohtQueues.current.get(ohtId);
        
            if (queue && queue.length > 0) {
                const updatedOHT = queue.shift()!;
                let oht = d3.select(`#oht-${updatedOHT.id}`);
        
                if (oht.empty()) {
                    oht = g.append('circle')
                        .attr('id', `oht-${updatedOHT.id}`)
                        .attr('class', 'oht')
                        .attr('cx', yScale(updatedOHT.x))
                        .attr('cy', yScale(updatedOHT.y))
                        .attr('r', 3)
                        .attr('fill', 'orange');
                }
        
                oht.transition()
                    .duration(25)
                    .attr('cx', yScale(updatedOHT.x))
                    .attr('cy', yScale(updatedOHT.y))
                    .on('end', () => {
                        if (queue.length > 0) {
                            processOHTQueue(ohtId);
                        } else {
                            processingOHTQueues.current.set(ohtId, false);
                        }
                    });
            } else {
                processingOHTQueues.current.set(ohtId, false);
            }
        };

        const processRailQueue = (edgeKey: string) => {
            const queue = edgeQueues.current.get(edgeKey);
        
            if (queue && queue.length > 0) {
                const updatedEdge = queue.shift()!;
                const rail = railsRef.current.find(
                    (r) => r.from === updatedEdge.from && r.to === updatedEdge.to
                );
        
                if (rail) {
                    rail.count = updatedEdge.count;
                    rail.avg_speed = updatedEdge.avg_speed;
                    updateRailColor(rail); // 색상 업데이트
                }
        
                if (queue.length > 0) {
                    processRailQueue(edgeKey);
                } else {
                    processingEdgeQueues.current.set(edgeKey, false);
                }
            } else {
                processingEdgeQueues.current.set(edgeKey, false);
            }
        };

        const processAllQueues = () => {

            // console.log('hihi');

            const firstOHTEntry = ohtQueues.current.entries().next().value;
            if (firstOHTEntry) {
                const [ohtId, queue] = firstOHTEntry;
        
                if (queue.length > 0) {
                    const time = queue[0].time; // 첫 번째 OHT의 time 확인

                    if (time == maxTime){
                        setIsRunning(false);
                    }
                }
            }
            
            // OHT 데이터 처리
            ohtQueues.current.forEach((queue, ohtId) => {
                if (queue.length > 0 && !processingOHTQueues.current.get(ohtId)) {
                    processingOHTQueues.current.set(ohtId, true);
                    processOHTQueue(ohtId);
                }
            });
        
            // Rail 데이터 처리
            edgeQueues.current.forEach((queue, edgeKey) => {
                if (queue.length > 0 && !processingEdgeQueues.current.get(edgeKey)) {
                    processingEdgeQueues.current.set(edgeKey, true);
                    processRailQueue(edgeKey);
                }
            });
        
            // 다음 프레임 요청
            requestAnimationFrame(processAllQueues);
        };
        
        
        socket.on('updateOHT', handleOHTUpdate);
        requestAnimationFrame(processAllQueues);


        return () => {
            socket.off('updateOHT', handleOHTUpdate);
            ohtQueues.current.clear();
            edgeQueues.current.clear();
            processingOHTQueues.current.clear();
            processingEdgeQueues.current.clear();
        };

    }, [data]);  // Remove railCounts from dependencies

    const modiRail = () => {
        if (selectedRail) {

            const removedRailKey =  `${selectedRail.rail.from}-${selectedRail.rail.to}`;

            const currentOHTPositions = Array.from(ohtQueues.current.entries()).map(([id, queue]) => queue[0]);
            const currentTime = currentOHTPositions.length > 0 ? currentOHTPositions[0].time : 0; 

            const isRemoved = d3.selectAll('.rail')
            .filter(d => d === selectedRail.rail)
            .classed('removed');

            d3.selectAll('.rail')
            .filter(d => d === selectedRail.rail)
            .attr('stroke', isRemoved ? colorScale(selectedRail.rail.count) : 'gray') // Restore or remove color
            .classed('removed', !isRemoved); 

            socket.disconnect();
            socket.connect();    

            ohtQueues.current.clear();
            edgeQueues.current.clear();
            processingOHTQueues.current.clear();
            processingEdgeQueues.current.clear();

            socket.emit('stopSimulation');

            socket.off('simulationStopped');


            socket.on('simulationStopped', () => {
                console.log('Simulation stopped confirmed by backend.');
            
                // Notify backend of the removed rail and current OHT positions
                socket.emit('modiRail', {
                    removedRailKey,
                    ohtPositions: currentOHTPositions,
                    currentTime,
                    isRemoved: !isRemoved,
                });
                setIsRunning(true);
                socket.off('simulationStopped');

            });

            setSelectedRail(null);
            setIsRunning(true);
        }
    };

    const startSimulation = () => {
        resetSimulation(); // Reset the state before starting
        console.log('Starting simulation');
        setIsRunning(true);
        socket.emit('startSimulation', { max_time: maxTime });
    };

    const resetSimulation = () => {
        console.log('Resetting simulation');
        
        // Reset rail counts and update colors
        d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; // Reset count directly on the data
            })
            .attr('stroke', d => colorScale(d.count))
            .classed('removed', false); // Reset color to default
    
        // Remove all OHT elements from the SVG
        d3.selectAll('.oht').remove();

        ohtQueues.current.clear();
        edgeQueues.current.clear();
        processingOHTQueues.current.clear();
        processingEdgeQueues.current.clear();

        socket.disconnect();
        socket.connect();

        ohtQueues.current.clear();
        edgeQueues.current.clear();
        processingOHTQueues.current.clear();
        processingEdgeQueues.current.clear();

        d3.selectAll('.rail')
        .each(function (d: Rail) {
            d.count = 0; // Reset count directly on the data
            d.avg_speed = 1500;
        })
        .attr('stroke', d => colorScale(d.count)); // Reset color to default

        d3.selectAll('.oht').remove();

        console.log('Simulation reset complete');
    };
    
    const stopSimulation = () => {
        console.log('Stopping simulation');
        setIsRunning(false);
        socket.emit('stopSimulation');
    };

    const computeButtonPosition = (x: number, y: number) => {
        const svgElement = svgRef.current;
        if (!svgElement) return { left: 0, top: 0 };
    
        // Get the bounding box of the SVG element
        const svgRect = svgElement.getBoundingClientRect();
    
        // Apply zoom and pan transformations
        const transform = zoomTransformRef.current;
    
        // Transform SVG coordinates to screen coordinates
        const transformedX = transform.x + x * transform.k;
        const transformedY = transform.y + y * transform.k;
    
        // Add the offset of the SVG element's position on the page
        return {
            left: svgRect.left + transformedX,
            top: svgRect.top + transformedY,
        };
    };
    

    const zoomIn = () => {
        const svg = d3.select(svgRef.current);
        svg.transition().call(zoomRef.current.scaleBy, 1.2);
    };

    const zoomOut = () => {
        const svg = d3.select(svgRef.current);
        svg.transition().call(zoomRef.current.scaleBy, 0.8);
    };

    return (
        <div className="flex flex-col h-screen" onClick={() => setSelectedRail(null)}>
            <header className="flex justify-between items-center p-4 bg-gray-800 text-white">
                <h1>OHT Railway Simulation</h1>
                <div className="flex gap-2">
                    <button 
                        className={`p-2 rounded ${displayMode === 'count' ? 'bg-blue-500' : 'bg-gray-500'}`}
                        onClick={() => setDisplayMode('count')}
                    >
                        Show Count
                    </button>
                    <button 
                        className={`p-2 rounded ${displayMode === 'avg_speed' ? 'bg-blue-500' : 'bg-gray-500'}`}
                        onClick={() => setDisplayMode('avg_speed')}
                    >
                        Show Avg Speed
                    </button>
                </div>
                <div className="flex gap-2">
                    <button className="w-10 h-10 bg-blue-500 text-white rounded hover:bg-blue-700 flex items-center justify-center" onClick={zoomIn}>+</button>
                    <button className="w-10 h-10 bg-blue-500 text-white rounded hover:bg-blue-700 flex items-center justify-center" onClick={zoomOut}>-</button>
                </div>
            </header>
            <main className="flex-grow">
                <div className="w-full h-full"
                    onClick={(e) => { // SVG 내부 클릭 시 유지
                        setSelectedRail(null); // 다른 곳 클릭 시 초기화
                    }}>
                    <svg ref={svgRef} id="oht-visualization" className="w-full h-full" onClick={(e) => setSelectedRail(null)}>
                        <g ref={gRef}></g>
                    </svg>
                    <div id="tooltip" className="tooltip" style={{ position: 'absolute', visibility: 'hidden', background: '#fff', border: '1px solid #ccc', padding: '5px', borderRadius: '5px', pointerEvents: 'none', fontSize: '10px' }}></div>
                    {selectedRail && (
                        <button
                            style={{
                                position: 'absolute',
                                ...computeButtonPosition(selectedRail.x, selectedRail.y),
                                transform: 'translate(20%, 0%)',
                                background: d3.selectAll('.rail')
                                .filter(d => d === selectedRail.rail)
                                .classed('removed') ? 'blue' : 'red',
                                color: 'white',
                                border: 'none',
                                borderRadius: '5px',
                                padding: '5px 10px',
                                cursor: 'pointer',
                                zIndex: 10,
                            }}
                            onClick={(e) => {
                                e.stopPropagation();
                                modiRail();
                            }}
                        >
                        {d3.selectAll('.rail')
                            .filter(d => d === selectedRail.rail)
                            .classed('removed') ? 'Restore Rail' : 'Remove Rail'}
                        </button>
                    )}
                </div>
            </main>
            <footer className="flex justify-between items-center p-4 bg-gray-800 text-white">
                <div className="flex gap-2">
                    <div className="flex flex-col items-start">
                        <label htmlFor="max-time-input" className="text-sm font-semibold mb-1">
                            Max Time:
                        </label>
                        <input
                            id="max-time-input"
                            type="number"
                            value={maxTime}
                            onChange={(e) => setMaxTime(Number(e.target.value))}
                            className="p-2 rounded border border-gray-300 focus:outline-none focus:ring focus:ring-blue-500 text-black w-28"
                        />
                    </div>
                    <button className="p-2 bg-blue-500 text-white rounded hover:bg-blue-700" onClick={() => {
                        if (!isRunning) {
                            resetSimulation();
                            startSimulation();
                        }
                        else {
                            stopSimulation();
                            resetSimulation();
                        }
                    }}>
                        {isRunning ? 'Stop Simulation' : 'Start Simulation'}
                    </button>
                </div>
            </footer>
        </div>
    );
};

export default OHTVisualization;
