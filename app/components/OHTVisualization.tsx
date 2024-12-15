import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';

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
            .domain([0, 100])
            .range([
                '#0000ff',
                '#ff0000'
                ]);

const OHTVisualization: React.FC<OHTVisualizationProps> = ({ data }) => {
    const [isRunning, setIsRunning] = useState(false);
    const svgRef = useRef<SVGSVGElement | null>(null);
    const gRef = useRef<SVGGElement | null>(null);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const zoomTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity.translate(100, 50).scale(0.5));
    const ohtQueues = useRef<Map<string, OHT[]>>(new Map()); // Use ohtQueues to track last positions
    const processingQueues = useRef<Map<string, boolean>>(new Map());
    const railsRef = useRef<Rail[]>(data.rails); // Maintain a reference to the rails


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
            .on('mouseout', hideTooltip);

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

        const updateRailColor = (railKey: string) => {
                const rail = rails.find(r => `${r.from}-${r.to}` === railKey);
                if (rail) {
                    g.selectAll('.rail')
                        .filter(d => `${d.from}-${d.to}` === railKey)
                        .transition()
                        .duration(500) // Transition duration for color change
                        .attr('stroke', colorScale(rail.count)); // Update stroke color based on count
                }
            };
            

        const processQueue = (ohtId: string) => {
            const queue = ohtQueues.current.get(ohtId);
        
            if (queue && queue.length > 0) {
                const updatedOHT = queue.shift()!;
                let oht = d3.select(`#oht-${updatedOHT.id}`);
        
                // If the OHT does not exist, create it
                if (oht.empty()) {
                    oht = g.append('circle')
                        .attr('id', `oht-${updatedOHT.id}`)
                        .attr('class', 'oht')
                        .attr('cx', yScale(updatedOHT.x))
                        .attr('cy', yScale(updatedOHT.y))
                        .attr('r', 2)
                        .attr('fill', 'orange');
                }                
        
                const lastKnownOHT = ohtQueues.current.get(updatedOHT.id)?.[0];
        
                // Animate the OHT to its new position
                oht.transition()
                    .duration(25) // Adjust animation duration
                    .attr('cx', yScale(updatedOHT.x))
                    .attr('cy', yScale(updatedOHT.y))
                    .on('start', () => {
                        // Check if the OHT has moved to a different rail
                        if (!lastKnownOHT || lastKnownOHT.source !== updatedOHT.source || lastKnownOHT.dest !== updatedOHT.dest) {
                            const key = `${updatedOHT.source}-${updatedOHT.dest}`;
                            const rail = rails.find(r => `${r.from}-${r.to}` === key);
                            if (rail) {
                                rail.count += 1; // Increment the count for this rail
                                updateRailColor(key); // Update rail color immediately
                            }
                        }
                    })
                    .on('end', () => {
                        if (queue.length > 0) {
                            requestAnimationFrame(() => processQueue(ohtId)); // Process the next position
                        } else {
                            processingQueues.current.set(ohtId, false);
                            checkSimulationComplete();
                        }
                    });
            } else {
                processingQueues.current.set(ohtId, false);
                checkSimulationComplete();
            }
        };
        
        
        const handleOHTUpdate = (data: { time: number; oht_positions: OHT[] }) => {
            data.oht_positions.forEach((updatedOHT) => {
                if (!ohtQueues.current.has(updatedOHT.id)) {
                    ohtQueues.current.set(updatedOHT.id, []);
                }
        
                // Add new position data to the queue for the OHT
                const queue = ohtQueues.current.get(updatedOHT.id)!;
                queue.push(updatedOHT);
        
                // Start processing if not already in progress
                if (!processingQueues.current.get(updatedOHT.id)) {
                    processingQueues.current.set(updatedOHT.id, true);
                    requestAnimationFrame(() => processQueue(updatedOHT.id));
                }
            });

            checkSimulationComplete();
        };

        const checkSimulationComplete = () => {
            // 모든 OHT 큐가 비어 있는지 확인
            const allQueuesEmpty = Array.from(ohtQueues.current.values()).every(queue => queue.length === 0);
        
            if (allQueuesEmpty) {
                console.log('Simulation complete: All OHT queues are empty.');
                setIsRunning(false); // 시뮬레이션 상태를 멈춤으로 변경
                socket.emit('simulationStopped'); // 백엔드에 시뮬레이션 완료 알림
            }
        };
        
        socket.on('updateOHT', handleOHTUpdate);

        return () => {
            socket.off('updateOHT', handleOHTUpdate);
        };

    }, [data]);  // Remove railCounts from dependencies

    const startSimulation = () => {
        resetSimulation(); // Reset the state before starting
        console.log('Starting simulation');
        setIsRunning(true);
        socket.emit('startSimulation');
    };

    const resetSimulation = () => {
        console.log('Resetting simulation');
        
        // Reset rail counts and update colors
        d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; // Reset count directly on the data
            })
            .attr('stroke', d => colorScale(d.count)); // Reset color to default
    
        // Remove all OHT elements from the SVG
        d3.selectAll('.oht').remove();
    
        // Clear OHT queues and processing states
        ohtQueues.current.clear();
        processingQueues.current.clear();
    
        console.log('Simulation reset complete');
    };
    
    const stopSimulation = () => {
        console.log('Stopping simulation');
        setIsRunning(false);
    
        // Disconnect and reconnect socket to clear data
        socket.disconnect();
        socket.connect();
    
        resetSimulation(); // Clear SVG elements and data
        socket.emit('stopSimulation');
    };

    // useEffect(() => {
    //     // Handle simulation stopped event
    //     socket.on('simulationStopped', () => {
    //         console.log('Simulation stopped confirmed by backend.');
    //         resetSimulation(); // Perform reset after backend confirms stop
    //     });
    
    //     return () => {
    //         socket.off('simulationStopped');
    //     };
    // }, []);
    

    const zoomIn = () => {
        const svg = d3.select(svgRef.current);
        svg.transition().call(zoomRef.current.scaleBy, 1.2);
    };

    const zoomOut = () => {
        const svg = d3.select(svgRef.current);
        svg.transition().call(zoomRef.current.scaleBy, 0.8);
    };

    return (
        <div className="flex flex-col h-screen">
            <header className="flex justify-between items-center p-4 bg-gray-800 text-white">
                <h1>OHT Railway Simulation</h1>
                <div className="flex gap-2">
                    <button className="w-10 h-10 bg-blue-500 text-white rounded hover:bg-blue-700 flex items-center justify-center" onClick={zoomIn}>+</button>
                    <button className="w-10 h-10 bg-blue-500 text-white rounded hover:bg-blue-700 flex items-center justify-center" onClick={zoomOut}>-</button>
                </div>
            </header>
            <main className="flex-grow">
                <div className="w-full h-full">
                    <svg ref={svgRef} id="oht-visualization" className="w-full h-full">
                        <g ref={gRef}></g>
                    </svg>
                    <div id="tooltip" className="tooltip" style={{ position: 'absolute', visibility: 'hidden', background: '#fff', border: '1px solid #ccc', padding: '5px', borderRadius: '5px', pointerEvents: 'none', fontSize: '10px' }}></div>
                </div>
            </main>
            <footer className="flex justify-between items-center p-4 bg-gray-800 text-white">
                <div className="flex gap-2">
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
