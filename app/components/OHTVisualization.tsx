import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';
import pako from 'pako'; // Gzip 압축 해제를 위해 pako 라이브러리 사용
import { SunIcon, MoonIcon } from "@heroicons/react/24/outline"; // Tailwind Heroicons 추가
import Modal from "./Modal"; // ✅ 모달 컴포넌트 추가 (결과 조회용)



const socket = io('http://localhost:5000');

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
    const [acceleratedTime, setAcceleratedTime] = useState(0);
    const [isAccelEnabled, setIsAccelEnabled] = useState(false);  // 가속 실험 여부

    const [isRunning, setIsRunning] = useState(false);
    const [isRunningBack, setIsRunningBack] = useState(false);


    const svgRef = useRef<SVGSVGElement | null>(null);
    const gRef = useRef<SVGGElement | null>(null);
    const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
    const zoomTransformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity.translate(100, 50).scale(0.5));

    const railsRef = useRef<Rail[]>(data.rails); // Maintain a reference to the rails
    const [selectedRail, setSelectedRail] = useState<{ rail: Rail; x: number; y: number } | null>(null);
    // const displayModeRef = useRef<'count' | 'avg_speed'>('count'); // useRef로 displayMode 관리
    const [updated, setUpdated] = useState(false);

    const simulTime = useRef(0);


    const [displayMode, setDisplayMode] = useState<'count' | 'avg_speed'>('count'); // 버튼 상태 관리
    const displayModeRef = useRef(displayMode);

    const lastOHTPositions = useRef<OHT[]>([]);

    const initialBufferSize = 100; // 초기 큐 크기 설정
    const isInitialBufferReady = useRef(false); // 초기 큐 준비 상태
    
    const lastEdgeStates = useRef<Map<string, Rail>>(new Map());
    const rafId = useRef(null);  // Add ref for RAF ID to cancel it

    const maxTimeref = useRef<HTMLInputElement | null>(null);
    const accTimeref = useRef<HTMLInputElement | null>(null);


    const [selectedJobFile, setSelectedJobFile] = useState<File | null>("");
    const jobFileInputRef = useRef<HTMLInputElement | null>(null); // useRef를 사용하여 파일 input 참조 생성

    const [selectedOhtFile, setSelectedOhtFile] = useState<File | null>("");
    const OhtFileInputRef = useRef<HTMLInputElement | null>(null); // useRef를 사용하여 파일 input 참조 생성

    const [isLoading, setIsLoading] = useState(false); // 스피너 상태 추가

    const ohtQueue: Array<{ time: number; updates: any[] }> = [];
    const edgeQueue: Array<{ time: number; updates: any[] }> = [];

    const ohtQueueRef = useRef<Array<{ time: number; updates: any[] }>>([]);
    const edgeQueueRef = useRef<Array<{ time: number; updates: any[] }>>([]);

    const yScaleRef = useRef<d3.ScaleLinear<number, number>>(d3.scaleLinear());

    const [darkMode, setDarkMode] = useState(() => {
        return localStorage.getItem("theme") === "dark";
    });

    const processTimeStepRef = useRef<() => void>(() => {});

    const [ohtMode, setOhtMode] = useState<"random" | "file">("random");


    const [ohtCount, setOhtCount] = useState(500);  // OHT 개수 상태 추가
    const ohtCountRef = useRef<HTMLInputElement | null>(null);  // OHT 개수 입력 필드 참조


    const [showModal, setShowModal] = useState(false); // ✅ 모달 상태 추가



    const handleTimeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setAcceleratedTime(Number(event.target.value));
    };

    const handleAccelChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setIsAccelEnabled(event.target.checked);  // 체크박스 상태에 따라 활성화 여부 결정
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files ? event.target.files[0] : null;
        if (event.target === jobFileInputRef.current) {
            setSelectedJobFile(file); // job 파일 설정
        } else if (event.target === OhtFileInputRef.current) {
            setSelectedOhtFile(file); // oht 파일 설정
        }
    };


    const objectToString = (obj: any) => {
        return Object.entries(obj).map(([key, value]) => `${key}: ${value}`).join('\n');
    };

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
            rail.avg_speed = 1500;
        });

        // Find max values for scaling
        const maxX = d3.max(nodes, d => d.x) || 1;
        const maxY = d3.max(nodes, d => d.y) || 1;

        // Create scales
        const yScale = d3.scaleLinear().domain([0, maxY]).range([0, 1200 - margin.top - margin.bottom]);

        yScaleRef.current = yScale;
        // Scale function for nodes and ports
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
            .attr('stroke', d => {
                // Calculate the initial color based on the current display mode
                const value = displayModeRef.current === 'count'
                    ? d.count / 100 // Normalize `count`
                    : (1500-d.avg_speed) / 500; // Normalize `avg_speed`
                return colorScale(Math.max(0, Math.min(1, value))); // Clamp value between 0 and 1
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

        
        const handleOHTUpdate = (data: { data: string }) => {
            const decompressedData = decompressData(data.data);
            if (!decompressedData) return;

            const { time, oht_positions, edges } = decompressedData;

            ohtQueueRef.current.push({ time, updates: oht_positions });
            edgeQueueRef.current.push({ time, updates: edges });

            // console.log(ohtQueue);

            if (!isInitialBufferReady.current) {
                if (ohtQueueRef.current.length >= initialBufferSize && edgeQueueRef.current.length >= initialBufferSize) {
                    isInitialBufferReady.current = true;
                }
            }
        };

        const getColorByStatus = (status: string) => {
            if (status === "STOP_AT_START") return "blue";
            if (status === "STOP_AT_END") return "red";
            return "orange";
        };


        processTimeStepRef.current = () => {

            if (!isInitialBufferReady.current) {
                rafId.current = requestAnimationFrame(processTimeStepRef.current);
                return;
            }

            setIsLoading(false);

            // console.log(edgeQueueRef.current);

            const ohtData = ohtQueueRef.current.shift(); // OHT 큐에서 데이터 꺼내기
            const edgeData = edgeQueueRef.current.shift(); // Edge 큐에서 데이터 꺼내기


            if (ohtQueueRef.current.length === 0 || edgeQueueRef.current.length === 0) {
                setTimeout(() => {
                    rafId.current = requestAnimationFrame(processTimeStepRef.current);
                }, 1000);
                return;
            }

            const { time: ohtTime, updates: ohtUpdates } = ohtData;
            const { time: edgeTime, updates: edgeUpdates } = edgeData;

            simulTime.current = ohtTime;

            lastOHTPositions.current = ohtUpdates; // OHT 상태를 저장
            
            edgeUpdates.forEach((updatedEdge: Rail) => {
                lastEdgeStates.current.set(`${updatedEdge.from}-${updatedEdge.to}`, updatedEdge); // Edge 상태를 Map에 저장
            });

            let pendingOHTTransitions = ohtUpdates.length;

            // Process OHTs for current time
            ohtUpdates.forEach((updatedOHT) => {
                const oht = d3.select(`#oht-${updatedOHT.id}`)
                            .data(ohtUpdates);

                if (oht.empty()) {
                    g.append("circle")
                        .attr("id", `oht-${updatedOHT.id}`)
                        .attr("class", "oht")
                        .attr("cx", yScale(updatedOHT.x))
                        .attr("cy", yScale(updatedOHT.y))
                        .attr("r", 5)
                        .attr("fill", getColorByStatus(updatedOHT.status));

                        pendingOHTTransitions--;                      
                        if (pendingOHTTransitions === 0) {
                            // When all OHT transitions for this time step are done, process edges
                            processEdgesForTime(edgeUpdates);
                        }
                }

                else{
                    oht.transition()
                    .duration(50)
                    .ease(d3.easeLinear) 
                    .attr("cx", yScale(updatedOHT.x))
                    .attr("cy", yScale(updatedOHT.y))
                    .attr("fill", getColorByStatus(updatedOHT.status))
                    .on("end", () => {
                        pendingOHTTransitions--;                      
                        if (pendingOHTTransitions === 0) {
                            // When all OHT transitions for this time step are done, process edges
                            processEdgesForTime(edgeUpdates);
                        }
                    });
                }
                });
                
            if (Number(maxTimeref.current.value) - ohtTime <= 0.3){
                setIsRunning(false);
                return;
            }


            // rafId.current = requestAnimationFrame(processTimeStepRef.current);
            };

        const processEdgesForTime = (edgeUpdates: any[]) => {
                edgeUpdates.forEach((updatedEdge) => {
                    const rail = railsRef.current.find(
                        (r) => r.from === updatedEdge.from && r.to === updatedEdge.to
                    );
        
                    if (rail) {
                        rail.count = updatedEdge.count;
                        rail.avg_speed = updatedEdge.avg_speed;
                        updateRailColor(rail); // Update rail color based on state
                    }
                });
                       
                rafId.current = requestAnimationFrame(processTimeStepRef.current);

            };
        
        socket.on('updateOHT', handleOHTUpdate);

        socket.on("backSimulationFinished", () => {
            setIsRunningBack(false);
        });
    

        // rafId.current = requestAnimationFrame(processTimeStep);

        return () => {
            socket.off('updateOHT', handleOHTUpdate);
            d3.selectAll('.oht').remove();
            ohtQueueRef.current = [];
            edgeQueueRef.current = [];

            cancelAnimationFrame(rafId.current);
            rafId.current = null;

        };

    }, [data]);  // Remove railCounts from dependencies

    const lightModeColors = {
        node: "red",
        rail: (d: Rail) => colorScale(Math.max(0, Math.min(1, displayModeRef.current === 'count' ? d.count / 100 : (1500 - d.avg_speed) / 500))),
        port: "green"
    };
    
    const darkModeColors = {
        node: "white",  // 토마토색
        rail: (d: Rail) => colorScale(Math.max(0, Math.min(1, displayModeRef.current === 'count' ? d.count / 100 : (1500 - d.avg_speed) / 500))).replace("rgb", "rgba").replace(")", ", 0.8)"),  // 반투명
        port: "#00ff7f"  // 밝은 녹색
    };
    
    // ✅ 다크모드 전환 시 색상 변경 함수
    const updateColors = () => {
        const colors = darkMode ? darkModeColors : lightModeColors;
    
        // Node 색상 업데이트
        d3.selectAll(".node")
            .transition().duration(500)
            .attr("fill", colors.node);
    
        // Rail 색상 업데이트
        d3.selectAll(".rail:not(.removed)")
            .transition().duration(500)
            .attr("stroke", d => colors.rail(d));

    
        // Port 색상 업데이트
        d3.selectAll(".port")
            .transition().duration(500)
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
    

    const updateRailColor = (rail: Rail) => {
        const selectedRail = d3.selectAll('.rail')
            .filter((d: Rail) => d.from === rail.from && d.to === rail.to)


        // if (!updated){
        //     selectedRail
        //     .data([rail]);
        //     setUpdated(true);
        // }
    
        // Check if the rail is marked as removed
        const isRemoved = selectedRail.classed('removed');
        if (isRemoved) {
            return; // Do not update color for removed rails
        }
    
        const value = displayModeRef.current === 'count' 
            ? rail.count / 100 // Normalize `count` (max 100)
            : (1500-rail.avg_speed) / 500; // Normalize `avg_speed` (max 1500)

        selectedRail.transition()
            .duration(25) // Smooth color change
            .attr('stroke', colorScale(Math.max(0, Math.min(1, value))))
    };

    const modiRail = () => {

        if (rafId.current) {
            cancelAnimationFrame(rafId.current);
            rafId.current = null;  // Make sure rafId.current is set before cancelling
        }

        setIsLoading(true);
        if (selectedRail) {
            const currentTime = simulTime.current; // 현재 시뮬레이션 시간을 가져오기
            const currentOHTPositions = lastOHTPositions.current; // 현재 OHT 상태
            const currentEdgeStates = Array.from(lastEdgeStates.current.values()); // 현재 Edge 상태를 Array로 변환
    
            const removedRailKey =  `${selectedRail.rail.from}-${selectedRail.rail.to}`;

            const isRemoved = d3.selectAll('.rail')
            .filter(d => d === selectedRail.rail)
            .classed('removed');

            d3.selectAll('.rail')
            .filter(d => d === selectedRail.rail)
            .attr('stroke', () => {
                if (isRemoved) {
                    // Restore color based on displayMode
                    const value = displayModeRef.current === 'count'
                        ? selectedRail.rail.count / 100 // Normalize count
                        : (1500-selectedRail.rail.avg_speed) / 500; // Normalize avg_speed
                    return colorScale(Math.max(0, Math.min(1, value))); // Clamp value
                }
                return 'gray'; // Set to gray when removed
            })
            .classed('removed', !isRemoved); 


            socket.disconnect();
            socket.connect();    



            isInitialBufferReady.current = false; // 초기 상태로 변경

            if (rafId.current) {
                cancelAnimationFrame(rafId.current);
                rafId.current = null; // 반드시 초기화
            }

            socket.emit('stopSimulation');

            socket.off('simulationStopped');

            socket.on('simulationStopped', () => {
                console.log('Simulation stopped confirmed by backend.');            
                // Notify backend of the removed rail and current OHT positions
                
                ohtQueueRef.current = [];
                edgeQueueRef.current = [];

                socket.emit('modiRail', {
                    removedRailKey,
                    ohtPositions: currentOHTPositions,
                    edges: currentEdgeStates,
                    currentTime,
                    isRemoved: !isRemoved,
                });

                // requestAnimationFrame(processTimeStep);
                setIsRunning(true);
                if (!rafId.current) {
                    rafId.current = requestAnimationFrame(processTimeStepRef.current);
                }    
                socket.off('simulationStopped');

            });

            setSelectedRail(null);
            setIsRunning(true);
        }
    };

    const startSimulation = () => {
        // resetSimulation(); // Reset the state before starting
        console.log('Starting simulation');
        setIsLoading(true);

        const formData = new FormData();

        formData.append('oht_file', selectedOhtFile);
        formData.append('job_file', selectedJobFile);

        socket.emit('uploadFiles', formData);  // 소켓을 통해 파일 전송

        socket.on('filesProcessed', (data) => {
            console.log('Files successfully uploaded:', data);

            maxTimeref.current.value = maxTime;
            
            const simulationData = { max_time: maxTime, num_OHTs: ohtCount };
            if (isAccelEnabled) {
                simulationData.current_time = acceleratedTime;  // current_time을 추가
            }
            socket.emit('startSimulation', simulationData);  // 시뮬레이션 시작 요청
    
            setSelectedJobFile(null);
            setSelectedOhtFile(null); // Reset the file input when starting the simulation
            if (jobFileInputRef.current) {
                jobFileInputRef.current.value = "";
            }
            
            if (OhtFileInputRef.current) {
                OhtFileInputRef.current.value = "";
            }
            setIsRunning(true);

            if (!rafId.current) {
                rafId.current = requestAnimationFrame(processTimeStepRef.current);
            }

            socket.off('filesProcessed');
        });
    
    };

    const startBackSimulation = () => {
        // resetSimulation(); // Reset the state before starting
        console.log('Starting simulation');
        setIsRunningBack(true);

        const formData = new FormData();

        formData.append('oht_file', selectedOhtFile);
        formData.append('job_file', selectedJobFile);

        socket.emit('uploadFiles', formData);  // 소켓을 통해 파일 전송

        socket.on('filesProcessed', (data) => {
            console.log('Files successfully uploaded:', data);

            maxTimeref.current.value = maxTime;
            
            const simulationData = { max_time: maxTime, num_OHTs: ohtCount };
            if (isAccelEnabled) {
                simulationData.current_time = acceleratedTime;  // current_time을 추가
            }
            socket.emit('onlySimulation', simulationData);  // 시뮬레이션 시작 요청
    
            setIsRunningBack(true);

            setSelectedJobFile(null);
            setSelectedOhtFile(null); // Reset the file input when starting the simulation
            if (jobFileInputRef.current) {
                jobFileInputRef.current.value = "";
            }
            
            if (OhtFileInputRef.current) {
                OhtFileInputRef.current.value = "";
            }

            if (!rafId.current) {
                rafId.current = requestAnimationFrame(processTimeStepRef.current);
            }

            socket.off('filesProcessed');
        });
    
    };

    const resetSimulation = () => {
        console.log('Resetting simulation');

        if (rafId.current) {
            cancelAnimationFrame(rafId.current);
            rafId.current = null;  // Make sure rafId.current is set before cancelling
        }
    
        // Remove all OHT elements from the SVG
        d3.selectAll('.oht').remove();

        isInitialBufferReady.current = false;

        socket.disconnect();
        socket.connect();

        ohtQueueRef.current = [];
        edgeQueueRef.current = [];

        // Reset rail counts and update colors
        railsRef.current.forEach((rail) => {
            rail.count = 0;
            rail.avg_speed = 1500; // ✅ avg_speed도 초기화
        });

        d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; // Reset count directly on the data
                d.avg_speed = 1500;
            })
            .classed('removed', false); // Reset color to default

        d3.selectAll('.rail')
        .each(function (d: Rail) {
            d.count = 0; // Reset count directly on the data
            d.avg_speed = 1500;
        })
        .attr('stroke', d => {
            // Calculate the initial color based on the current display mode
            const value = displayModeRef.current === 'count'
                ? d.count / 100 // Normalize `count`
                : (1500-d.avg_speed) / 500; // Normalize `avg_speed`
            return colorScale(Math.max(0, Math.min(1, value))); // Clamp value between 0 and 1
        }); // Reset color to default

        d3.selectAll('.oht').remove();
        setIsLoading(false);
        
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

                        // Reset rail counts and update colors
            railsRef.current.forEach((rail) => {
                rail.count = 0;
                rail.avg_speed = 1500; // ✅ avg_speed도 초기화
            });

            d3.selectAll('.rail')
                .each(function (d: Rail) {
                    d.count = 0; // Reset count directly on the data
                    d.avg_speed = 1500;
                })
                .classed('removed', false); // Reset color to default
   
            d3.selectAll('.rail')
            .each(function (d: Rail) {
                d.count = 0; // Reset count directly on the data
                d.avg_speed = 1500;
            })
            .attr('stroke', d => {
                // Calculate the initial color based on the current display mode
                const value = displayModeRef.current === 'count'
                    ? d.count / 100 // Normalize `count`
                    : (1500-d.avg_speed) / 500; // Normalize `avg_speed`
                return colorScale(Math.max(0, Math.min(1, value))); // Clamp value between 0 and 1
            }); // Reset color to default
    
            socket.off('simulationStopped');

            socket.disconnect();
            socket.connect();
        });

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
        <div className={`flex flex-col h-screen ${darkMode ? "bg-[#0F172A] text-gray-200" : "bg-white text-gray-900"}`} onClick={() => setSelectedRail(null)}>
            {/* 헤더 */}
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


            {/* 메인 시각화 */}
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
                            background: darkMode ? '#1E293B' : '#fff',  // ✅ 다크모드 반영
                            color: darkMode ? '#fff' : '#000',         // ✅ 텍스트 색상 변경
                            border: `1px solid ${darkMode ? '#475569' : '#ccc'}`, // ✅ 테두리 색 변경
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
            </main>

            {/* 푸터 컨트롤 UI */}
            <footer className={`flex flex-col md:flex-row items-center justify-between p-4 ${darkMode ? "bg-[#1E293B]" : "bg-[#E2E8F0]"} shadow-lg`}>
                <div className="flex flex-col md:flex-row gap-6 items-center">
                    {/* ✅ OHT 모드 선택 */}
                    <div className="flex flex-col items-center gap-4">
                        <span className="text-sm font-semibold">OHT Mode</span>
                        
                        {/* 랜덤 모드 선택 */}
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

                        {/* 파일 업로드 모드 선택 */}
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

                    {/* ✅ 랜덤 모드에서만 OHT 개수 입력 가능 */}


                    {/* ✅ 가속 활성화 */}
                    <div className="flex items-center gap-3">
                        <input
                            type="checkbox"
                            checked={isAccelEnabled}
                            onChange={handleAccelChange}
                            className="h-5 w-5 rounded border-gray-400 text-blue-600 focus:ring focus:ring-blue-400"
                        />
                        <label className="text-sm font-medium">Enable Acceleration</label>
                    </div>

                    {/* ✅ 가속 시간 입력 */}
                    {isAccelEnabled && (
                        <div className="flex flex-col items-center gap-3">
                            <label htmlFor="accel-time-input" className="text-sm font-semibold">
                                Acceleration Time
                            </label>
                            <input
                                ref={accTimeref}
                                id="accel-time-input"
                                type="number"
                                value={acceleratedTime}
                                onChange={handleTimeChange}
                                className={`p-2 rounded-md border ${darkMode ? "border-gray-600 bg-gray-700 text-white" : "border-gray-400 bg-white text-black"} 
                                focus:outline-none focus:ring focus:ring-blue-500 w-32 text-center`}
                                placeholder="Enter time"
                            />
                        </div>
                    )}

                    {/* ✅ 최대 시간 입력 */}
                    <div className="flex flex-col items-center gap-3">
                        <label htmlFor="max-time-input" className="text-sm font-semibold">
                            Max Time
                        </label>
                        <input
                            ref={maxTimeref}
                            id="max-time-input"
                            type="number"
                            value={maxTime}
                            onChange={(e) => setMaxTime(Number(e.target.value))}
                            className={`p-2 rounded-md border ${darkMode ? "border-gray-600 bg-gray-700 text-white" : "border-gray-400 bg-white text-black"} 
                            focus:outline-none focus:ring focus:ring-blue-500 w-32 text-center`}
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

                {/* ✅ 파일 업로드 UI (파일 모드에서만 표시) */}
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
                {/* ✅ 시뮬레이션 시작 / 정지 버튼 */}
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
                {/* ✅ 시뮬레이션 시작 / 정지 버튼 */}
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
