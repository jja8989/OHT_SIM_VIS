import React, { useEffect, useState, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';
import pako from 'pako';
import { SunIcon, MoonIcon } from "@heroicons/react/24/outline"; 
import Modal from "./Modal"; 
import { getClientId } from '../utils/getClientId';
import SimulationControls from "./SimulationControls";
import TimeInput from "./TimeInput"; // 불러오기


const client_id = getClientId();


const socket = io(process.env.NEXT_PUBLIC_SOCKET_URL || '/', {
    path: '/socket.io',
    transports: ['websocket'],
    query: {
        client_id: client_id,
      }
  });

interface Node {
    id: string;
    x: number;
    y: number;
}

interface Rail {
    from: string;
    to: string;
    count: number;
    max_speed : number;
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
    time: number;
    source: string;
    dest: string;
    status: string;
}

interface LayoutData {
    nodes: Node[];
    rails: Rail[];
    ports: Port[];
}

interface OHTVisualizationProps {
    data: LayoutData;
}

const colorScale = d3.scaleLinear<string, string>()
    .domain([0, 1])
    .range([
        '#0000ff',
        '#ff0000'
    ]);

const decompressData = (compressedData: string) => {
    try {
        const decodedData = atob(compressedData);
        const byteArray = new Uint8Array(decodedData.split('').map(char => char.charCodeAt(0)));
        const jsonData = pako.inflate(byteArray, { to: 'string' });
        return JSON.parse(jsonData);
    } catch (error) {
        console.error('Error decompressing data:', error);
        return null;
    }
};
                

const OHTVisualization: React.FC<OHTVisualizationProps> = ({ data }) => {
    const [maxTime, setMaxTime] = useState(3600);
    const [acceleratedTime, setAcceleratedTime] = useState(0);
    const [isAccelEnabled, setIsAccelEnabled] = useState(false);

    const [isRunning, setIsRunning] = useState(false);
    const [isRunningBack, setIsRunningBack] = useState(false);


    const svgRef = useRef<SVGSVGElement | null>(null);
    const gRef = useRef<SVGGElement | null>(null);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const zoomTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity.translate(100, 50).scale(0.5));

    const railsRef = useRef<Rail[]>(data.rails);
    const [selectedRail, setSelectedRail] = useState<{ rail: Rail; x: number; y: number } | null>(null);

    const stopAtRef = useRef<number>(maxTime);
    const simulTime = useRef(0);


    const [displayMode, setDisplayMode] = useState<'count' | 'avg_speed'>('count');
    const displayModeRef = useRef(displayMode);

    const lastOHTPositions = useRef<OHT[]>([]);
    
    const lastEdgeStates = useRef<Map<string, Rail>>(new Map());
    const rafId = useRef(null);  

    const maxTimeref = useRef<HTMLInputElement | null>(null);
    const accTimeref = useRef<HTMLInputElement | null>(null);


    const [selectedJobFile, setSelectedJobFile] = useState<File | null>("");
    const jobFileInputRef = useRef<HTMLInputElement | null>(null); 

    const [selectedOhtFile, setSelectedOhtFile] = useState<File | null>("");
    const OhtFileInputRef = useRef<HTMLInputElement | null>(null); 

    const [isLoading, setIsLoading] = useState(false); 

    const ohtQueueRef = useRef<Array<{ time: number; updates: any[] }>>([]);
    const edgeQueueRef = useRef<Array<{ updates: any[] }>>([]);

    const yScaleRef = useRef<d3.ScaleLinear<number, number>>(d3.scaleLinear());

    const [darkMode, setDarkMode] = useState(() => {
        return localStorage.getItem("theme") === "dark";
    });

    const processTimeStepRef = useRef<() => void>(() => {});

    const [ohtMode, setOhtMode] = useState<"random" | "file">("random");


    const [ohtCount, setOhtCount] = useState(500);
    const [showModal, setShowModal] = useState(false);

    const [isPlaying, setIsPlaying] = useState(false);
    const [speedMultiplier, setSpeedMultiplier] = useState(1);
    const speedMultiplierRef = useRef(speedMultiplier);


    const railNodeMapRef = useRef<Map<string, SVGLineElement>>(new Map());
    const railDataMapRef = useRef<Map<string, Rail>>(new Map());

    const speeds = [0.1, 0.25, 0.33, 0.5, 1, 2, 3, 4]; 
    const [speedIndex, setSpeedIndex] = useState(4); 

    const [currentSimulTime, setCurrentSimulTime] = useState(0);




    function formatTime(secs: number) {
    const h = String(Math.floor(secs / 3600)).padStart(2, "0");
    const m = String(Math.floor((secs % 3600) / 60)).padStart(2, "0");
    const s = String(Math.floor(secs % 60)).padStart(2, "0");
    return `${h}:${m}:${s}`;
    }
    const handleAccelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsAccelEnabled(event.target.checked);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files ? event.target.files[0] : null;
        if (event.target === jobFileInputRef.current) {
            setSelectedJobFile(file);
        } else if (event.target === OhtFileInputRef.current) {
            setSelectedOhtFile(file);
        }
    };


    const objectToString = (obj: any) => {
        return Object.entries(obj).map(([key, value]) => `${key}: ${value}`).join('\n');
    };


    const play = () => {
    
    d3.selectAll(".oht").interrupt();

    if (!rafId.current) {
        rafId.current = requestAnimationFrame(processTimeStepRef.current);
    }
    setIsPlaying(true);
    };

    const pause = () => {
    d3.selectAll(".oht").interrupt();

    if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
    }

    setIsPlaying(false);
    };
    

    const faster = () => {
        setSpeedIndex(prev => Math.min(prev + 1, speeds.length - 1))
        d3.selectAll(".oht").interrupt();
        setSpeedMultiplier(speeds[speedIndex])
        if (!rafId.current) {
            rafId.current = requestAnimationFrame(processTimeStepRef.current);
        }
    };
    const slower = () => {
        setSpeedIndex(prev => Math.max(prev - 1, 0))
        d3.selectAll(".oht").interrupt();
        setSpeedMultiplier(speeds[speedIndex])
        if (!rafId.current) {
            rafId.current = requestAnimationFrame(processTimeStepRef.current);
        }
    };


    const trianglePath = useMemo(() => {
        const base = 10;
        const height = 14; // 좀 더 뾰족하게
        return `M 0 -${height/2} L ${base/2} ${height/2} L -${base/2} ${height/2} Z`;
        }, []);

    
    useEffect(() => {

        const svg = d3.select(svgRef.current)
            .attr('width', '100%')
            .attr('height', '100%');

        const margin = { top: 50, right: 50, bottom: 50, left: 50 };

        const g = d3.select(gRef.current);

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.5, 5])
            .translateExtent([[-100, -100], [2400, 1300]])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
                zoomTransformRef.current = event.transform;
            });

        svg.call(zoom).call(zoom.transform, zoomTransformRef.current);
        zoomRef.current = zoom;

        const { nodes, rails, ports } = data;

        rails.forEach(rail => {
            rail.count = 0;
            rail.avg_speed = rail.max_speed;
        });

        const maxY = d3.max(nodes, d => d.y) || 1;

        const yScale = d3.scaleLinear().domain([0, maxY]).range([0, 1200 - margin.top - margin.bottom]);

        yScaleRef.current = yScale;

        const scalePosition = (d: { x: number; y: number }) => ({
            x: yScale(d.x),
            y: yScale(d.y)
        });


        const tooltip = d3.select('#tooltip');

        const showTooltip = (event: MouseEvent, content: string) => {
            tooltip.style('visibility', 'visible')
                .style('left', `${event.pageX + 10}px`)
                .style('top', `${event.pageY - 100}px`)
                .html(content.replace(/\n/g, '<br>'));
        };

        const hideTooltip = () => {
            tooltip.style('visibility', 'hidden');
        };

        g.selectAll('.rail')
            .data(rails)
            .enter()
            .append('line')
            .attr('class', 'rail')
            .attr('x1', d => scalePosition(nodes.find(n => n.id === d.from)!).x)
            .attr('y1', d => scalePosition(nodes.find(n => n.id === d.from)!).y)
            .attr('x2', d => scalePosition(nodes.find(n => n.id === d.to)!).x)
            .attr('y2', d => scalePosition(nodes.find(n => n.id === d.to)!).y)
            .attr('stroke-width', 2.5)
            .attr('stroke', d => {
                const value = displayModeRef.current === 'count'
                    ? d.count / 100
                    : (d.max_speed-d.avg_speed) / d.max_speed;
                return colorScale(Math.max(0, Math.min(1, value)));
            })
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
            })
            .each(function(d: Rail) {
                const key = `${d.from}-${d.to}`;

                railNodeMapRef.current.set(key, this as SVGLineElement);
                railDataMapRef.current.set(key, d);

            });

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

        
        const handleOHTUpdate = (data: { data: string }) => {
            const decompressedData = decompressData(data.data);
            if (!decompressedData) return;

            const { time, oht_positions, edges } = decompressedData;

            ohtQueueRef.current.push({ time, updates: oht_positions });
            edgeQueueRef.current.push({ updates: edges });

            setIsLoading(false);
        };

        processTimeStepRef.current = () => {
                const ohtData = ohtQueueRef.current.shift();
                const edgeData = edgeQueueRef.current.shift();                    

                if (!ohtData || !edgeData) {
                    rafId.current = requestAnimationFrame(processTimeStepRef.current);
                    return;
                }

                setIsLoading(false);

                const { time: ohtTime, updates: ohtUpdates } = ohtData;
                const { updates: edgeUpdates } = edgeData;


                let pending = ohtUpdates.length;

                const commitStep = () => {
                    for (const u of (edgeUpdates ?? [])) {
                    const key = `${u.from}-${u.to}`;
                    const rail_data    = railDataMapRef.current.get(key);
                    const rail_segment = railNodeMapRef.current.get(key);
                    if (!rail_data || !rail_segment) continue;

                    rail_data.count     = u.count;
                    rail_data.avg_speed = u.avg_speed;

                    const sel = d3.select(rail_segment);
                    if (!sel.classed('removed')) {
                        const value = (displayModeRef.current === 'count')
                        ? rail_data.count / 100
                        : (rail_data.max_speed - rail_data.avg_speed) / rail_data.max_speed;
                        rail_segment.setAttribute('stroke', colorScale(Math.max(0, Math.min(1, value))));
                    }
                    }

                    simulTime.current = ohtTime;
                    lastOHTPositions.current = ohtUpdates;
                    edgeUpdates.forEach((updatedEdge: Rail) => {
                    lastEdgeStates.current.set(`${updatedEdge.from}-${updatedEdge.to}`, updatedEdge);
                    });
                    rafId.current = requestAnimationFrame(processTimeStepRef.current);
                };

                const doneOne = () => {
                    if (--pending === 0) commitStep();
                };

                ohtUpdates.forEach((updatedOHT) => {
                    const oht = d3.select(`#oht-${updatedOHT.id}`);

                    const cx = yScale(updatedOHT.x);
                    const cy = yScale(updatedOHT.y);

                    if (oht.empty()) {
                        g.append("path")
                            .attr("id", `oht-${updatedOHT.id}`)
                            .attr("class", "oht")
                            .attr("d", trianglePath)
                            .attr("fill", getColorByStatus(updatedOHT.status))
                            .attr("transform", `translate(${cx},${cy}) rotate(${updatedOHT.angleDeg})`);
                            doneOne();

                        } 
                        else{
                        oht.transition()
                            .duration(200 / speedMultiplierRef.current)
                            .ease(d3.easeLinear)
                            .attr("fill", getColorByStatus(updatedOHT.status))
                            .attr("transform", `translate(${cx},${cy}) rotate(${updatedOHT.angleDeg})`)
                            .on("end interrupt", doneOne);
                        }
                });

                // isAfterCut.current = false;

                if (stopAtRef.current - simulTime.current <= 1) {
                    setIsRunning(false);
                    setIsPlaying(false);
                    if (rafId.current) { cancelAnimationFrame(rafId.current); rafId.current = null; }
                    return;
                }
                };
        
        socket.on('updateOHT', handleOHTUpdate);

        socket.on("backSimulationFinished", () => {
            console.log('back ended')

            setIsRunningBack(false);
        });
    
        return () => {
            socket.off('updateOHT', handleOHTUpdate);
            d3.selectAll('.oht').remove();
            ohtQueueRef.current = [];
            edgeQueueRef.current = [];

            cancelAnimationFrame(rafId.current);
            rafId.current = null;
            socket.off("backSimulationFinished");

        };

    },[data]);


    const getColorByStatus = (status: string) => {
        if (status === "STOP_AT_START") return "blue";
        if (status === "STOP_AT_END") return "red";
        return "orange";
    };


    const lightModeColors = {
        node: "red",
        rail: (d: Rail) => colorScale(Math.max(0, Math.min(1, displayModeRef.current === 'count' ? d.count / 100 : (d.max_speed - d.avg_speed) / d.max_speed))),
        port: "green"
    };
    
    const darkModeColors = {
        node: "white",  
        rail: (d: Rail) => colorScale(Math.max(0, Math.min(1, displayModeRef.current === 'count' ? d.count / 100 : (d.max_speed - d.avg_speed) / d.max_speed))).replace("rgb", "rgba").replace(")", ", 0.8)"),  // 반투명
        port: "#00ff7f" 
    };
    
    const updateColors = () => {
        const colors = darkMode ? darkModeColors : lightModeColors;

        d3.selectAll(".node")
            .attr("fill", colors.node);

        d3.selectAll(".rail:not(.removed)")
            .attr("stroke", d => colors.rail(d));

        d3.selectAll(".port")
            .attr("fill", colors.port);
    };
    

    useEffect(() => {
        if (darkMode) {
            document.documentElement.classList.add("dark");
            localStorage.setItem("theme", "dark");
        } else {
            document.documentElement.classList.remove("dark");
            localStorage.setItem("theme", "light");
        }

        updateColors();
    }, [darkMode]);

    useEffect(() => {
        speedMultiplierRef.current = speedMultiplier;
    }, [speedMultiplier]);

    useEffect(() => {
        if (!isRunning) {
            setCurrentSimulTime(0); // 실행 멈추면 바로 0 세팅
            return;
        }

        const interval = setInterval(() => {
            setCurrentSimulTime(simulTime.current);
        }, 500);

        return () => clearInterval(interval);
        }, [isRunning]);
    

    const updateRailColor = (rail: Rail) => {
        const key = `${rail.from}-${rail.to}`;
        const node = railNodeMapRef.current.get(key);
        if (!node) return;

        const sel = d3.select(node);
        if (sel.classed('removed')) {
            node.setAttribute('stroke', 'gray');
            return;
        }

        const value = displayModeRef.current === 'count'
            ? rail.count / 100
            : (rail.max_speed - rail.avg_speed) / rail.max_speed;

        const nextColor = colorScale(Math.max(0, Math.min(1, value)));

        node.setAttribute('stroke', nextColor);
        };

    const modiRail = () => {
        if (rafId.current) {
            cancelAnimationFrame(rafId.current);
            rafId.current = null;
        }

        setIsLoading(true);


        if (selectedRail) {

            socket.disconnect();
            
            ohtQueueRef.current = [];
            edgeQueueRef.current = [];

            const currentTime = simulTime.current;
            const currentOHTPositions = lastOHTPositions.current;
            const currentEdgeStates = Array.from(lastEdgeStates.current.values());
            const removedRailKey =  `${selectedRail.rail.from}-${selectedRail.rail.to}`;

            const railElement = railNodeMapRef.current.get(removedRailKey);

            const sel = d3.select(railElement);

            const isRemoved = sel.classed("removed");

            sel
            .attr('stroke', () => {
                if (isRemoved) {
                    const value = displayModeRef.current === 'count'
                        ? selectedRail.rail.count / 100
                        : (selectedRail.rail.max_speed-selectedRail.rail.avg_speed) / selectedRail.rail.max_speed; 
                    return colorScale(Math.max(0, Math.min(1, value))); 
                }
                return 'gray';
            })
            .classed('removed', !isRemoved);

            socket.once('connect', () => {
                socket.emit('stopSimulation');
            });
            
            socket.connect();

            socket.off('simulationStopped');

            socket.on('simulationStopped', () => {
                console.log('Simulation stopped confirmed by backend.');            

                
                ohtQueueRef.current = [];
                edgeQueueRef.current = [];
                
                socket.emit('modiRail', {
                    removedRailKey,
                    ohtPositions: currentOHTPositions,
                    edges: currentEdgeStates,
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

    const startSimulation = async () => {

        console.log('Starting simulation');
        setIsLoading(true);

        let jobBuffer = null;   
        let ohtBuffer = null;

        if (selectedJobFile) {
            jobBuffer = await selectedJobFile.arrayBuffer();
        }

        if (selectedOhtFile) {
            ohtBuffer = await selectedOhtFile.arrayBuffer();
        }

        socket.emit('uploadFiles', {
            job_file: jobBuffer,
            oht_file: ohtBuffer
        });


        socket.on('filesProcessed', (data) => {
            console.log('Files successfully uploaded:', data);

            maxTimeref.current.value = maxTime;
            stopAtRef.current = maxTime;
            
            const simulationData = { max_time: maxTime, num_OHTs: ohtCount };
            if (isAccelEnabled) {
                simulationData.current_time = acceleratedTime; 
            }
            socket.emit('startSimulation', simulationData); 
    
            setSelectedJobFile(null);
            setSelectedOhtFile(null);
            if (jobFileInputRef.current) {
                jobFileInputRef.current.value = "";
            }
            
            if (OhtFileInputRef.current) {
                OhtFileInputRef.current.value = "";
            }
            setIsRunning(true);
            setIsPlaying(true);



            if (!rafId.current) {
                rafId.current = requestAnimationFrame(processTimeStepRef.current);
            }

            socket.off('filesProcessed');
        });
    
    };

    const startBackSimulation = async () => {
        console.log('Starting simulation');
        setIsRunningBack(true);

        let jobBuffer = null;   
        let ohtBuffer = null;

        if (selectedJobFile) {
            jobBuffer = await selectedJobFile.arrayBuffer();
        }

        if (selectedOhtFile) {
            ohtBuffer = await selectedOhtFile.arrayBuffer();
        }

        socket.emit('uploadFiles', {
            job_file: jobBuffer,
            oht_file: ohtBuffer
        });

        socket.on('filesProcessed', (data) => {
            console.log('Files successfully uploaded:', data);

            maxTimeref.current.value = maxTime;
            
            const simulationData = { max_time: maxTime, num_OHTs: ohtCount };
            if (isAccelEnabled) {
                simulationData.current_time = acceleratedTime;
            }
            socket.emit('onlySimulation', simulationData); 
    
            setIsRunningBack(true);

            setSelectedJobFile(null);
            setSelectedOhtFile(null); 
            if (jobFileInputRef.current) {
                jobFileInputRef.current.value = "";
            }
            
            if (OhtFileInputRef.current) {
                OhtFileInputRef.current.value = "";
            }

            socket.off('filesProcessed');
        });
    
    };

    const resetSimulation = () => {
        console.log('Resetting simulation');

        if (rafId.current) {
            cancelAnimationFrame(rafId.current);
            rafId.current = null; 
        }
    

        d3.selectAll('.oht').remove();

        socket.disconnect();
        socket.connect();

        ohtQueueRef.current = [];
        edgeQueueRef.current = [];

        railsRef.current.forEach((rail) => {
            rail.count = 0;
            rail.avg_speed = rail.max_speed;
        });

        d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; 
                d.avg_speed = d.max_speed;
            })
            .classed('removed', false); 

        d3.selectAll('.rail')
        .each(function (d: Rail) {
            d.count = 0; 
            d.avg_speed = d.max_speed;
        })
        .attr('stroke', d => {
            const value = displayModeRef.current === 'count'
                ? d.count / 100
                : (d.max_speed-d.avg_speed) / d.max_speed;
            return colorScale(Math.max(0, Math.min(1, value))); 
        });

        d3.selectAll('.oht').remove();
        setIsLoading(false);
        setIsPlaying(false);
        setSpeedIndex(4);
        setSpeedMultiplier(speeds[speedIndex]);
        simulTime.current = 0;
        setCurrentSimulTime(0);
        
        console.log('Simulation reset complete');
    };
    
    const stopSimulation = () => {
        console.log('Stopping simulation');
        setIsRunning(false);
        socket.emit('stopSimulation');

        socket.off('simulationStopped');

        setIsLoading(false);

        socket.on('simulationStopped', () => {
            d3.selectAll('.oht').remove();

            ohtQueueRef.current = [];
            edgeQueueRef.current = [];

            railsRef.current.forEach((rail) => {
                rail.count = 0;
                rail.avg_speed = rail.max_speed;
            });

            d3.selectAll('.rail')
                .each(function (d: Rail) {
                    d.count = 0;
                    d.avg_speed = d.max_speed;
                })
                .classed('removed', false);
   
            d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; 
                d.avg_speed = d.max_speed;
            })
            .attr('stroke', d => {

                const value = displayModeRef.current === 'count'
                    ? d.count / 100 
                    : (d.max_speed-d.avg_speed) / d.max_speed; 
                return colorScale(Math.max(0, Math.min(1, value))); 
            });
    
            socket.off('simulationStopped');

        });
        simulTime.current = 0;
        setCurrentSimulTime(0);

    };


    const stopBackSimulation = () => {
        socket.emit('stopBackSimulation');

        setIsRunningBack(false);


        socket.on('simulationBackStopped', () => {
            setIsRunningBack(false);
            socket.off('simulationBackStopped');
        });

    };

    const computeButtonPosition = (x: number, y: number) => {
        const svgElement = svgRef.current;
        if (!svgElement) return { left: 0, top: 0 };
    

        const svgRect = svgElement.getBoundingClientRect();
        const transform = zoomTransformRef.current;
    
        const transformedX = transform.x + x * transform.k;
        const transformedY = transform.y + y * transform.k;
    
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
        <div className={`flex flex-col h-screen ${darkMode ? "bg-[#0F172A] text-gray-200" : "bg-white text-gray-900"}`} onClick={() => setSelectedRail(null)}>
            <header className={`flex justify-between items-center p-4 ${darkMode ? "bg-[#1E293B]" : "bg-[#F8FAFC]"} shadow-md`}>
            <h1 className="text-lg font-semibold tracking-wide">OHT Railway Simulation</h1>
                <div className="flex gap-2">
                    <button 
                        className={`p-2 rounded ${displayMode === 'count' ? 'bg-blue-600' : 'bg-gray-600'} transition hover:bg-blue-700`}
                        onClick={() => {
                            displayModeRef.current = 'count';
                            setDisplayMode('count');
                            railsRef.current.forEach(updateRailColor);
                        }}
                    >
                        Show Count
                    </button>
                    <button 
                        className={`p-2 rounded ${displayMode === 'avg_speed' ? 'bg-blue-600' : 'bg-gray-600'} transition hover:bg-blue-700`}
                        onClick={() => {
                            displayModeRef.current = 'avg_speed';
                            setDisplayMode('avg_speed');
                            railsRef.current.forEach(updateRailColor);
                        }}
                    >
                        Show Avg Speed
                    </button>
                </div>
                <div className="flex gap-2">
                    <button className="w-10 h-10 bg-blue-600 text-white rounded-full hover:bg-blue-800 flex items-center justify-center shadow-md" onClick={zoomIn}>+</button>
                    <button className="w-10 h-10 bg-blue-600 text-white rounded-full hover:bg-blue-800 flex items-center justify-center shadow-md" onClick={zoomOut}>-</button>

                    <button 
                        className="p-2 bg-gray-700 text-white rounded hover:bg-gray-500 transition"
                        onClick={() => setShowModal(true)}
                        >
                            View Simulations
                        </button>

                    <button 
                        className="p-2 rounded-full bg-gray-600 hover:bg-gray-500 transition"
                        onClick={() => setDarkMode(!darkMode)}
                    >
                        {darkMode ? <SunIcon className="w-6 h-6 text-yellow-400" /> : <MoonIcon className="w-6 h-6 text-gray-900" />}
                    </button>
                </div>
            </header>

            <main className="flex-grow relative">
                {isLoading && (
                    <div className="absolute inset-0 flex justify-center items-center bg-gray-700 bg-opacity-80 z-10">
                    <div className="w-16 h-16 border-4 border-gray-300 border-opacity-50 border-t-blue-500 border-t-4 rounded-full animate-spin"></div>
                </div>
                )}
                <div className="w-full h-full" onClick={() => setSelectedRail(null)}>
                    <svg ref={svgRef} id="oht-visualization" className="w-full h-full">
                        <g ref={gRef}></g>
                    </svg>

                    <div id="tooltip" 
                        className="tooltip"
                        style={{
                            position: 'absolute', 
                            visibility: 'hidden',
                            background: darkMode ? '#1E293B' : '#fff',
                            color: darkMode ? '#fff' : '#000',        
                            border: `1px solid ${darkMode ? '#475569' : '#ccc'}`, 
                            padding: '5px', 
                            borderRadius: '5px', 
                            pointerEvents: 'none', 
                            fontSize: '10px'
                        }}>
                    </div>

                    {selectedRail && (
                        <button
                            style={{
                                position: 'absolute',
                                ...computeButtonPosition(selectedRail.x, selectedRail.y),
                                transform: 'translate(20%, 0%)',
                                background: d3.selectAll('.rail')
                                    .filter(d => d === selectedRail.rail)
                                    .classed('removed') ? '#2563EB' : '#DC2626',
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
                   <div className="absolute bottom-4 right-4 z-50">
  <div className="flex flex-col items-center gap-3 px-3 py-3
                  bg-transparent
                  border border-gray-300/40 dark:border-gray-600/40
                  rounded-md shadow-sm
                  text-xs">
    <SimulationControls
      isPlaying={isPlaying}
      onPlay={play}
      onPause={pause}
      onFaster={faster}
      onSlower={slower}
    />

    <div className="flex items-center gap-1">
      <span className="text-gray-600 dark:text-gray-300">⚡</span>
      <span className="font-mono text-sm text-gray-900 dark:text-white">
        x{speeds[speedIndex]}
      </span>
    </div>

    <div className="flex flex-col items-center gap-2 mt-2">
      <div className="flex items-center gap-1">
        <span className="text-gray-600 dark:text-gray-300">⏱</span>
        <span className="font-mono text-sm text-gray-900 dark:text-white">
          {formatTime(currentSimulTime)}
        </span>
      </div>
    </div>
  </div>
</div>

            </main>
            <footer className={`flex flex-col md:flex-row items-center justify-between p-4 ${darkMode ? "bg-[#1E293B]" : "bg-[#E2E8F0]"} shadow-lg`}>

           
                <div className="flex flex-col md:flex-row gap-6 items-center">
                    <div className="flex flex-col items-center gap-4">
                        <span className="text-sm font-semibold">OHT Mode</span>
                        <label className="flex items-center cursor-pointer">
                            <input
                                type="radio"
                                name="ohtMode"
                                value="random"
                                checked={ohtMode === "random"}
                                onChange={() => setOhtMode("random")}
                                className="hidden"
                            />
                            <span className={`px-3 py-1 rounded-lg transition text-sm font-medium cursor-pointer
                                ${ohtMode === "random" ? "bg-blue-600 text-white" : "bg-gray-300 dark:bg-gray-600 text-black dark:text-white"}
                            `}>
                                Random
                            </span>
                        </label>

                        <label className="flex items-center cursor-pointer">
                            <input
                                type="radio"
                                name="ohtMode"
                                value="file"
                                checked={ohtMode === "file"}
                                onChange={() => setOhtMode("file")}
                                className="hidden"
                            />
                            <span className={`px-3 py-1 rounded-lg transition text-sm font-medium cursor-pointer
                                ${ohtMode === "file" ? "bg-blue-600 text-white" : "bg-gray-300 dark:bg-gray-600 text-black dark:text-white"}
                            `}>
                                File Upload
                            </span>
                        </label>
                    </div>


                    <div className="flex items-center gap-3">
                        <input
                            type="checkbox"
                            checked={isAccelEnabled}
                            onChange={handleAccelChange}
                            className="h-5 w-5 rounded border-gray-400 text-blue-600 focus:ring focus:ring-blue-400"
                        />
                        <label className="text-sm font-medium">Enable Acceleration</label>
                    </div>


                    {isAccelEnabled && (
                        <div className="flex flex-col items-center gap-3">
                        <label className="text-sm font-semibold">Acceleration Time</label>
                        <TimeInput
                            ref={accTimeref}
                            value={acceleratedTime}
                            onChange={setAcceleratedTime}
                            className={
                            darkMode
                                ? "border-gray-600 bg-gray-700 text-white"
                                : "border-gray-400 bg-white text-black"
                            }
                        />
                        </div>
                    )}

                    <div className="flex flex-col items-center gap-3">
                        <label className="text-sm font-semibold">Max Time</label>
                        <TimeInput
                        ref={maxTimeref}
                        value={maxTime}
                        onChange={setMaxTime}
                        className={
                            darkMode
                            ? "border-gray-600 bg-gray-700 text-white"
                            : "border-gray-400 bg-white text-black"
                        }
                        />
                    </div>

                    
                {ohtMode === "random" && (
                        <div className="flex flex-col items-center gap-3">
                            <label htmlFor="oht-count-input" className="text-sm font-semibold">
                                Number of OHTs
                            </label>
                            <input
                                id="oht-count-input"
                                type="number"
                                value={ohtCount}
                                onChange={(e) => setOhtCount(Number(e.target.value))}
                                className={`p-2 rounded-md border ${darkMode ? "border-gray-600 bg-gray-700 text-white" : "border-gray-400 bg-white text-black"} 
                                focus:outline-none focus:ring focus:ring-blue-500 w-32 text-center`}
                            />
                        </div>
                    )}
                    
 
                </div>


                {ohtMode === "file" && (
                    <div className="flex flex-col md:flex-row gap-6 items-center mt-4">
                        <div className="flex flex-col items-center">
                            <label className={`flex flex-col items-center px-4 py-2 ${darkMode ? "bg-blue-700 hover:bg-blue-800" : "bg-blue-500 hover:bg-blue-600"} text-white rounded-lg shadow-md transition cursor-pointer`}>
                                📂 Upload Job File
                                <input ref={jobFileInputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
                            </label>
                            {selectedJobFile ? (
                                <p className="text-sm text-green-400 mt-2">{selectedJobFile.name}</p>
                            ) : (
                                <p className="text-sm text-gray-400 mt-2">No file selected</p>
                            )}
                        </div>
                        <div className="flex flex-col items-center">
                            <label className={`flex flex-col items-center px-4 py-2 ${darkMode ? "bg-blue-700 hover:bg-blue-800" : "bg-blue-500 hover:bg-blue-600"} text-white rounded-lg shadow-md transition cursor-pointer`}>
                                📂 Upload OHT File
                                <input ref={OhtFileInputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
                            </label>
                            {selectedOhtFile ? (
                                <p className="text-sm text-green-400 mt-2">{selectedOhtFile.name}</p>
                            ) : (
                                <p className="text-sm text-gray-400 mt-2">No file selected</p>
                            )}
                        </div>
                    </div>
                )}

                <div className="flex flex-col md:flex-row gap-6 items-center mt-4">

                <button
                    className={`px-6 py-3 rounded-lg shadow-md transition text-white ${
                        isRunningBack ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
                    }`}
                    onClick={() => {
                        if (!isRunningBack) {
                            startBackSimulation();
                        } else {
                            stopBackSimulation();
                        }
                    }}
                >
                    {isRunningBack ? "Stop Simulation Only" : "Start Simulation Only"}
                </button>

                <button
                    className={`px-6 py-3 rounded-lg shadow-md transition text-white ${
                        isRunning ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600"
                    }`}
                    onClick={() => {
                        if (!isRunning) {
                            resetSimulation();
                            startSimulation();
                        } else {
                            resetSimulation();
                            stopSimulation();
                            resetSimulation();

                        }
                    }}
                >
                    {isRunning ? "Stop Simulation" : "Start Simulation"}
                </button>
                </div>
            </footer>



            {showModal && <Modal onClose={() => setShowModal(false)} />}
        </div>
    );
};

export default OHTVisualization;
