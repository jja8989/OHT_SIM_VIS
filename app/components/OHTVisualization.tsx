import React, { useEffect, useState, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import io from 'socket.io-client';
import pako from 'pako';
import { SunIcon, MoonIcon } from "@heroicons/react/24/outline"; 
import Modal from "./Modal"; 
import { getClientId } from '../utils/getClientId';
import SimulationControls from "./SimulationControls";
import TimeInput from "./TimeInput"; 

const client_id = getClientId();


const socket = io('/', {
    path: '/socket.io',
    transports: ['polling', 'websocket'],
    query: { client_id },
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
    pred_30?: number;
    pred_60?: number;
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

interface EdgeQueueItem {
  updates: any[];
  pred?: Record<string, number[]>; 
}


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

    const [displayMode, setDisplayMode] = useState<'count' | 'avg_speed' | 'pred30' | 'pred60'>('count');
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
    const edgeQueueRef = useRef<EdgeQueueItem[]>([]);

    const yScaleRef = useRef<d3.ScaleLinear<number, number>>(d3.scaleLinear());

    const [darkMode, setDarkMode] = useState(() => {
        return localStorage.getItem("theme") === "dark";
    });

    const processTimeStepRef = useRef<() => void>(() => {});

    const [ohtMode, setOhtMode] = useState<"random" | "file">("random");


    const [ohtCount, setOhtCount] = useState(500);
    const [showModal, setShowModal] = useState(false);

    const [isPlaying, setIsPlaying] = useState(false);
    const isPlayingRef = useRef(isPlaying);

    const [speedMultiplier, setSpeedMultiplier] = useState(1);
    const speedMultiplierRef = useRef(speedMultiplier);


    const railNodeMapRef = useRef<Map<string, SVGLineElement>>(new Map());
    const railDataMapRef = useRef<Map<string, Rail>>(new Map());

    const speeds = [0.1, 0.25, 0.33, 0.5, 1, 2, 3, 4]; 
    const [speedIndex, setSpeedIndex] = useState(4); 

    const BASE_STEP_MS = 100; 

    const computeStride = (multiplier: number) =>
        Math.max(1, Math.floor(multiplier)); 

    const computeDuration = (multiplier: number) =>
        multiplier >= 1 ? BASE_STEP_MS     
                        : BASE_STEP_MS / Math.max(1e-3, multiplier);

    const [currentSimulTime, setCurrentSimulTime] = useState(0);

    const headerRef = useRef<HTMLElement | null>(null);
    const footerRef = useRef<HTMLElement | null>(null);


    const edgeHistoriesRef = useRef<Map<
        string,
        Array<{ time: number; avg: number; p30?: number; p60?: number }>
        >>(new Map());


    const hoveredKeyRef = useRef<string | null>(null);
    const miniChartRef = useRef<HTMLDivElement | null>(null);



    const colorScale = useMemo(() => {
    return d3.scaleLinear<string, string>()
        .domain([0, 1])
        .range(
        darkMode
            ? ["#3399ff", "#ff6666"]
            : ["#0000ff", "#ff0000"]
        );
    }, [darkMode]);


    const colorScaleRef = useRef(colorScale);

    useEffect(() => {
        colorScaleRef.current = colorScale;
    }, [colorScale]);


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

    type ObjFmtOpts = {
    exclude?: string[];                    
    decimals?: number;                    
    fixedByKey?: Record<string, number>;  
    };

    const objectToString = (obj: Record<string, any>, opts: ObjFmtOpts = {}) => {
    const exclude = new Set(opts.exclude ?? []);
    const decimals = opts.decimals ?? 2;
    const fixedByKey = { count: 0, ...(opts.fixedByKey ?? {}) };

    return Object.entries(obj)
        .filter(([k, v]) => v !== undefined && v !== null && !exclude.has(k))
        .map(([k, v]) => {
        if (typeof v === 'number') {
            const places = fixedByKey[k] ?? decimals;
            return `${k}: ${v.toFixed(places)}`;
        }
        return `${k}: ${v}`;
        })
        .join('\n');
    };


    const play = () => {
    d3.selectAll(".oht").interrupt();

    setIsPlaying(true);
    
    if (!rafId.current) {
        rafId.current = requestAnimationFrame(processTimeStepRef.current);
    }

    };

    const pause = () => {
    d3.selectAll(".oht").interrupt();
    setIsPlaying(false);

    if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
    }


    };

    const faster = () => {

        setSpeedIndex(prev => {
            const next = Math.min(prev + 1, speeds.length - 1);
            setSpeedMultiplier(speeds[next]);
            return next;
        });
        };

        const slower = () => {

        setSpeedIndex(prev => {
            const next = Math.max(prev - 1, 0);
            setSpeedMultiplier(speeds[next]);
            return next;
        });
        };


    const trianglePath = useMemo(() => {
        const base = 8;
        const height = 10;
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

        const tooltip = d3.select('#tooltip')
            .style('position', 'absolute')
            .style('visibility', 'hidden')
            .style('z-index', '9999')    
            .style('pointer-events', 'none')
            .style('overflow', 'visible')
            .style('background', darkMode ? 'rgba(15,23,42,0.9)' : 'rgba(255,255,255,0.95)')
            .style('color', darkMode ? '#e5e7eb' : '#111827')
            .style('border', `1px solid ${darkMode ? '#334155' : '#cbd5e1'}`)
            .style('border-radius', '6px')
            .style('padding', '8px 10px');

        if (tooltip.select('#tip-info').empty()) {
            tooltip.html('');
            tooltip.append('div')
                .attr('id', 'tip-info')
                .style('font-size', '12px')
                .style('line-height', '1.3');

            tooltip.append('div')
            .attr('id', 'mini-chart')
            .style('margin-top', '10px')
            .style('width', '270px')
            .style('height', '200px')
            .style('background', darkMode ? 'rgba(255,255,255,0.95)' : 'rgba(0,0,0,0.75)')
            .style('color', darkMode ? '#111827' : '#e5e7eb')
            .style('border-radius', '4px')
            .style('padding', '4px');

            miniChartRef.current = document.getElementById('mini-chart') as HTMLDivElement;
            } else {
            miniChartRef.current = document.getElementById('mini-chart') as HTMLDivElement;
            }

        const showTooltip = (event: MouseEvent, content: string) => {
            tooltip
                .style('visibility', 'visible')
                .style('left', `${event.pageX + 12}px`)
                .style('top', `${event.pageY - 120}px`);

            tooltip.select('#tip-info').html(content.replace(/\n/g, '<br>'));
            tooltip.select('#mini-chart').style('visibility', 'visible');
            };

        const hideTooltip = () => {
        tooltip.style('visibility', 'hidden');
        tooltip.select('#mini-chart').style('visibility', 'hidden');
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
            .on('mouseover', (event, d) => {
                showTooltip(event, objectToString(d, { exclude: ['curve'] }))
                  const key = `${d.from}-${d.to}`;
                    hoveredKeyRef.current = key;

                    drawEdgeMiniChart(key);
                    d3.select(miniChartRef.current).style('visibility', 'visible');
              
            })
            .on('mouseout', () => {
                const chart = miniChartRef.current!;
                chart.style.visibility = 'hidden';
                hoveredKeyRef.current = null;
                hideTooltip();
                })
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
            .attr('fill', 'var(--node-color)')
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
            .attr('fill', 'var(--port-color)')
            .on('mouseover', (event, d) => showTooltip(event, objectToString(d, { exclude: ['x', 'y', 'distance'] })))
            .on('mouseout', hideTooltip);

        
        const handleOHTUpdate = (data: { data: string }) => {
            const decompressedData = decompressData(data.data);
            if (!decompressedData) return;

            const { time, oht_positions, edges, pred } = decompressedData;

            ohtQueueRef.current.push({ time, updates: oht_positions });
            edgeQueueRef.current.push({ updates: edges, pred });

            setIsLoading(false);
        };

        processTimeStepRef.current = () => {
            if (!isPlayingRef.current){
                return;
            }
            
                const multiplier = speedMultiplierRef.current;
                const stride = computeStride(multiplier);
                const durationMs = computeDuration(multiplier);

                let skip = Math.max(0, stride - 1);

                while (
                    skip > 0 &&
                    ohtQueueRef.current.length > 1 &&
                    edgeQueueRef.current.length > 1
                ) {
                    ohtQueueRef.current.shift(); 
                    const edgeData = edgeQueueRef.current.shift();
                    if (edgeData) {
                        const { pred } = edgeData;

                        // pred만 기록
                        if (pred) {
                        for (const key in pred["30"] || {}) {
                            const hist = edgeHistoriesRef.current.get(key);
                            if (hist && hist.length) {
                            hist[hist.length - 1].p30 = pred["30"]?.[key];
                            }
                        }
                        for (const key in pred["60"] || {}) {
                            const hist = edgeHistoriesRef.current.get(key);
                            if (hist && hist.length) {
                            hist[hist.length - 1].p60 = pred["60"]?.[key];
                            }
                        }
                        }
                    }

                    skip--;
                }

                const ohtData = ohtQueueRef.current.shift();
                const edgeData = edgeQueueRef.current.shift();                    

                if (!ohtData || !edgeData) {
                    rafId.current = requestAnimationFrame(processTimeStepRef.current);
                    return;
                }

                setIsLoading(false);

                const { time: ohtTime, updates: ohtUpdates } = ohtData;
                const { updates: edgeUpdates, pred } = edgeData;


                let pending = ohtUpdates.length;

                const commitStep = () => {
                    for (const u of (edgeUpdates ?? [])) {
                        const key = `${u.from}-${u.to}`;
                        const rail_data    = railDataMapRef.current.get(key);
                        const rail_segment = railNodeMapRef.current.get(key);
                        if (!rail_data || !rail_segment) continue;

                        rail_data.count     = u.count;
                        rail_data.avg_speed = u.avg_speed;

                        if (pred) {
                            if (pred["30"]) rail_data.pred_30 = pred["30"][key];
                            if (pred["60"]) rail_data.pred_60 = pred["60"][key];
                            }

                        const sel = d3.select(rail_segment);

                        if (!sel.classed("removed")) {
                        let value: number;

                        if (displayModeRef.current === "count") {
                            value = rail_data.count / 100;
                        } else if (displayModeRef.current === "avg_speed") {
                            value = (rail_data.max_speed - rail_data.avg_speed) / rail_data.max_speed;
                        } else if (displayModeRef.current === "pred30") {
                            value = (rail_data.max_speed - (rail_data.pred_30 ?? rail_data.avg_speed)) / rail_data.max_speed;
                        } else if (displayModeRef.current === "pred60") {
                            value = (rail_data.max_speed - (rail_data.pred_60 ?? rail_data.avg_speed)) / rail_data.max_speed;
                        } else {
                            value = 0;
                        }

                        const nextColor = colorScaleRef.current(Math.max(0, Math.min(1, value)));
                        rail_segment.setAttribute("stroke", nextColor);
                        }

                        const histKey = key;
                        const map = edgeHistoriesRef.current;
                        let hist = map.get(histKey) ?? [];
                        const nowSec = Math.floor(ohtTime); 


                        const last = hist[hist.length - 1];

                        if (!last || last.time < nowSec) {
                        hist.push({
                            time: nowSec,
                            avg: rail_data.avg_speed,
                            p30: last?.p30,
                            p60: last?.p60,
                        });
                        }

                        if (pred) {
                        const p30 = pred["30"]?.[key];
                        const p60 = pred["60"]?.[key];
                        if (p30 !== undefined) {
                            hist[hist.length - 1].p30 = p30;
                        }
                        if (p60 !== undefined) {
                            hist[hist.length - 1].p60 = p60;
                        }
                        }

                        const cutoff = nowSec - 120;
                        if (hist.length && hist[0].time < cutoff) {
                            hist = hist.filter(d => d.time >= cutoff);
                        }
                        map.set(histKey, hist);

                    }


                    if (hoveredKeyRef.current) {
                    drawEdgeMiniChart(hoveredKeyRef.current);
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
                            .duration(durationMs)           
                            .ease(d3.easeLinear)
                            .attr("fill", getColorByStatus(updatedOHT.status))
                            .attr("transform", `translate(${cx},${cy}) rotate(${updatedOHT.angleDeg})`)
                            .on("end", doneOne)
                            .on("interrupt", doneOne);

                        }
                });

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

    function drawEdgeMiniChart(key: string) {
        const wrap = miniChartRef.current;
        if (!wrap) return;

        const history = edgeHistoriesRef.current.get(key) ?? [];
        if (!history.length) {
            d3.select('#mini-chart').style('visibility', 'hidden');
            return;
        }
        d3.select('#mini-chart').style('visibility', 'visible');

        const isDark =
            document.body.classList.contains("dark") ||
            localStorage.getItem("theme") === "dark";
        const axisTextColor = isDark ? "#334155" : "#e5e7eb";
        const axisLineColor = "#94a3b8";

        const W = 270, H = 180;
        const m = { t: 30, r: 10, b: 36, l: 46 };
        const innerW = W - m.l - m.r;
        const innerH = H - m.t - m.b;

        let svg = d3.select(wrap).select<SVGSVGElement>("svg");
        const firstDraw = svg.empty();

        if (firstDraw) {
            svg = d3.select(wrap).append("svg")
                .attr("width", W).attr("height", H)
                .style("overflow", "visible");

            const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);

            g.append("g").attr("class", "x-axis").attr("transform", `translate(0,${innerH})`);
            g.append("g").attr("class", "y-axis");

            const graphGroup = g.append("g")
                .attr("class", "graph-group")
                .attr("clip-path", "url(#mini-clip)");

            graphGroup.append("path").attr("class", "line-avg")
                .attr("fill", "none").attr("stroke", "#60A5FA").attr("stroke-width", 2);
            graphGroup.append("path").attr("class", "line-p30")
                .attr("fill", "none").attr("stroke", "#F59E0B").attr("stroke-width", 2);
            graphGroup.append("path").attr("class", "line-p60")
                .attr("fill", "none").attr("stroke", "#34D399").attr("stroke-width", 2);
            graphGroup.append("g").attr("class", "last-dots");

            // 범례
            const legend = [
                { label: "avg", color: "#60A5FA" },
                { label: "pred30", color: "#F59E0B" },
                { label: "pred60", color: "#34D399" },
            ];
            const lg = g.append("g").attr("class","legend").attr("transform", `translate(0,-10)`);
            legend.forEach((l, i) => {
                lg.append("line")
                    .attr("x1", 70 * i).attr("x2", 70 * i + 22)
                    .attr("y1", 0).attr("y2", 0)
                    .attr("stroke", l.color).attr("stroke-width", 2);
                lg.append("text")
                    .attr("x", 70 * i + 26).attr("y", 4)
                    .attr("font-size", 10).attr("fill", axisTextColor)
                    .text(l.label);
            });
        }

        const g = svg.select<SVGGElement>("g");

        const last = history[history.length - 1];
        const now = last ? last.time : 0;
        const tMin = Math.max(0, now - 60);
        const tMax = now + 60;

        const clippedAvg  = history.filter(d => d.time >= tMin && d.time <= tMax);
        const clippedP30  = history.filter(d => d.p30 !== undefined && (d.time) >= tMin - 30 && (d.time) <= tMax);
        const clippedP60  = history.filter(d => d.p60 !== undefined && (d.time) >= tMin -60 && (d.time) <= tMax);

        const x = d3.scaleLinear().domain([tMin, tMax]).range([0, innerW]);
        const maxSpeed = d3.max([...clippedAvg, ...clippedP30, ...clippedP60],
                                d => Math.max(d.avg ?? 0, d.p30 ?? 0, d.p60 ?? 0)) || 1;
        const y = d3.scaleLinear().domain([0, maxSpeed]).nice().range([innerH, 0]);

        const minLabelPitch = 60;
        const maxTicks = Math.max(2, Math.floor(innerW / minLabelPitch));
        const tickValues = d3.ticks(tMin, tMax, maxTicks);
        const xAxis = d3.axisBottom(x)
            .tickValues(tickValues)
            .tickFormat((v: any) => {
                const secs = Number(v);
                const m = Math.floor(secs / 60);
                const s = secs % 60;
                return `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
            });

        g.select<SVGGElement>(".x-axis")
            .transition().duration(300)
            .call(xAxis as any)
            .selection()
            .selectAll("text").attr("fill", axisTextColor).style("font-size", 10);
        g.select<SVGGElement>(".x-axis").selectAll("path,line")
            .attr("stroke", axisLineColor).attr("opacity", 0.85);

        const yAxis = d3.axisLeft(y).ticks(4).tickFormat(d3.format(".1f"));
        g.select<SVGGElement>(".y-axis")
            .transition().duration(300)
            .call(yAxis as any)
            .selection()
            .selectAll("text").attr("fill", axisTextColor).style("font-size", 10);
        g.select<SVGGElement>(".y-axis").selectAll("path,line")
            .attr("stroke", axisLineColor).attr("opacity", 0.85);        

        const lineAvg = d3.line<{time:number;avg:number}>()
            .x(d => x(d.time)).y(d => y(d.avg));
        const lineP30 = d3.line<{time:number;p30?:number}>()
            .x(d => x(d.time + 30)).y(d => y(d.p30 as number));
        const lineP60 = d3.line<{time:number;p60?:number}>()
            .x(d => x(d.time + 60)).y(d => y(d.p60 as number));

        const morph = (sel: d3.Selection<SVGPathElement, unknown, any, any>, gen: any, data: any) => {
            const prevD = sel.attr("d") || "";
            const nextD = gen(data) || "";
            if (firstDraw || !prevD) {
                sel.attr("d", nextD);
            } else if (prevD !== nextD) {
                sel.transition().duration(260).attrTween("d", () => {
                    const i = d3.interpolateString(prevD, nextD);
                    return (t: number) => i(t);
                });
            }
        };

        morph(g.select<SVGPathElement>(".line-avg"), lineAvg, clippedAvg);
        morph(g.select<SVGPathElement>(".line-p30"), lineP30, clippedP30);
        morph(g.select<SVGPathElement>(".line-p60"), lineP60, clippedP60);

        const dotsData = [
            {v:last.avg,  cls:"avg-dot",  color:"#60A5FA", tx:last.time},
            {v:last.p30,  cls:"p30-dot",  color:"#F59E0B", tx:last.time + 30},
            {v:last.p60,  cls:"p60-dot",  color:"#34D399", tx:last.time + 60},
        ].filter(d => d.v !== undefined && d.tx >= tMin && d.tx <= tMax);

        const dots = g.select<SVGGElement>(".last-dots")
            .selectAll<SVGCircleElement, any>("circle")
            .data(dotsData, (d: any) => d.cls);

        dots.enter().append("circle")
            .attr("r", 0).attr("fill", d => d.color)
            .attr("cx", d => x(d.tx)).attr("cy", d => y(d.v))
            .transition().duration(220).attr("r", 2.5);

        dots.transition().duration(260)
            .attr("cx", d => x(d.tx)).attr("cy", d => y(d.v));

        dots.exit().transition().duration(200).attr("r", 0).remove();
    }


    const repaintRailsForTheme = () => {
        requestAnimationFrame(() => {
            railNodeMapRef.current.forEach((el, key) => {
            const rail = railDataMapRef.current.get(key);
            if (!rail) return;

            if (d3.select(el).classed('removed')) return;

            const value = displayModeRef.current === 'count'
                ? rail.count / 100
                : (rail.max_speed - rail.avg_speed) / rail.max_speed;

            el.setAttribute('stroke', colorScale(Math.max(0, Math.min(1, value))));
            });
        });
    };
    

    useEffect(() => {
        const svg = d3.select(svgRef.current);
        svg.style("background-color", darkMode ? "#0f172a" : "");


        if (darkMode) {
            headerRef.current?.classList.add("dark");
            footerRef.current?.classList.add("dark");
            

            d3.selectAll(".node").attr("fill", "white"); 


            localStorage.setItem("theme", "dark");
        } else {
            headerRef.current?.classList.remove("dark");
            footerRef.current?.classList.remove("dark");



            d3.selectAll(".node").attr("fill", "red"); 
            localStorage.setItem("theme", "light");
        }
        repaintRailsForTheme()

    }, [darkMode]);



    useEffect(() => {
        if (!isRunning) {
            setCurrentSimulTime(0); 
            return;
        }

        const interval = setInterval(() => {
            setCurrentSimulTime(simulTime.current);
        }, 500);

        return () => clearInterval(interval);
        }, [isRunning]);

    useEffect(() => {
        speedMultiplierRef.current = speedMultiplier;
    }, [speedMultiplier]);

    useEffect(() => {
        isPlayingRef.current = isPlaying;
    }, [isPlaying]);


    const modiRail = () => {
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

            if (isRemoved){
                selectedRail.rail.avg_speed = 0;
            }
            
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
        edgeHistoriesRef.current.clear();

        railsRef.current.forEach((rail) => {
            rail.count = 0;
            rail.avg_speed = rail.max_speed;
            rail.pred_30 = rail.max_speed;
            rail.pred_60 = rail.max_speed;
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

        if (miniChartRef.current) {
            d3.select(miniChartRef.current).selectAll("*").remove();
            miniChartRef.current.style.visibility = "hidden";
        }

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
        <div
            className="flex flex-col h-screen bg-white text-gray-900"
            onClick={() => setSelectedRail(null)}
        >
            <header ref={headerRef} className="flex justify-between items-center p-4 bg-[#F8FAFC] shadow-md header">
            <h1 className="text-lg font-semibold tracking-wide">OHT Railway Network Simulation</h1>

            <div className="flex gap-2">
                <button
                    className={`p-2 rounded transition ${displayMode === "count" ? "bg-blue-600 text-white" : "bg-gray-600 text-white"}`}
                    onClick={() => {
                        setDisplayMode("count")
                        displayModeRef.current = "count"
                    }}
                >
                    Count
                </button>

                <button
                    className={`p-2 rounded transition ${displayMode === "avg_speed" ? "bg-blue-600 text-white" : "bg-gray-600 text-white"}`}
                    onClick={() => {
                        setDisplayMode("avg_speed")
                        displayModeRef.current = "avg_speed"
                    }
                    }
                >
                    Avg Speed
                </button>

                <button
                    className={`p-2 rounded transition ${displayMode === "pred30" ? "bg-blue-600 text-white" : "bg-gray-600 text-white"}`}
                    onClick={() => {
                        setDisplayMode("pred30")
                        displayModeRef.current = "pred30"
                    }
                    }
                >
                    Pred 30s
                </button>

                <button
                    className={`p-2 rounded transition ${displayMode === "pred60" ? "bg-blue-600 text-white" : "bg-gray-600 text-white"}`}
                    onClick={() => {
                        setDisplayMode("pred60")
                        displayModeRef.current = "pred60"
                    }
                    }
                >
                    Pred 60s
                </button>
            </div>

            <div className="flex gap-2">
                <button
                className="w-10 h-10 bg-blue-600 text-white rounded-full hover:bg-blue-800 flex items-center justify-center shadow-md"
                onClick={zoomIn}
                >
                +
                </button>
                <button
                className="w-10 h-10 bg-blue-600 text-white rounded-full hover:bg-blue-800 flex items-center justify-center shadow-md"
                onClick={zoomOut}
                >
                -
                </button>

                <button
                className="p-2 bg-blue-600 text-white rounded hover:bg-blue-800 transition"
                onClick={() => setShowModal(true)}
                >
                View Simulations
                </button>

                <button
                className="p-2 rounded-full bg-gray-600 hover:bg-gray-500 transition"
                onClick={() => setDarkMode(!darkMode)}
                >
                {darkMode ? (
                    <SunIcon className="w-6 h-6 text-yellow-400" />
                ) : (
                    <MoonIcon className="w-6 h-6 text-gray-900" />
                )}
                </button>
            </div>
            </header>

            <main className="flex-grow relative">
            {isLoading && (
                <div className="absolute inset-0 flex justify-center items-center bg-gray-700/80 z-10">
                <div className="w-16 h-16 border-4 border-gray-300/50 border-t-blue-500 rounded-full animate-spin" />
                </div>
            )}

            <div className="w-full h-full" onClick={() => setSelectedRail(null)}>
                <svg ref={svgRef} id="oht-visualization" className="w-full h-full">
                    <g ref={gRef}></g>
                </svg>

                <div id="tooltip" className="tooltip" />

                {selectedRail && (
                    <button
                        style={{
                        position: "absolute",
                        ...computeButtonPosition(selectedRail.x, selectedRail.y),
                        transform: "translate(20%, 0%)",
                        background: d3
                            .selectAll(".rail")
                            .filter((d: any) => d === selectedRail.rail)
                            .classed("removed")
                            ? "#2563EB"
                            : "#DC2626",
                        color: "white",
                        border: "none",
                        borderRadius: "5px",
                        padding: "5px 10px",
                        cursor: "pointer",
                        zIndex: 10,
                        }}
                        onClick={(e) => {
                        e.stopPropagation();
                        modiRail();
                        }}
                    >

                        {d3
                        .selectAll(".rail")
                        .filter((d: any) => d === selectedRail.rail)
                        .classed("removed")
                        ? "Restore Rail"
                        : "Remove Rail"}
                    </button>
                    )}
            </div>

            <div className="absolute bottom-4 right-4 z-50">
                <div
                className="flex flex-col items-center gap-3 px-3 py-3
                            bg-transparent
                            border border-gray-300/40
                            rounded-md shadow-sm
                            text-xs"
                >
                <SimulationControls
                    isPlaying={isPlaying}
                    onPlay={play}
                    onPause={pause}
                    onFaster={faster}
                    onSlower={slower}
                />

                <div className="flex items-center gap-1">
                    <span className="text-gray-600">⚡</span>
                    <span className={`font-mono text-sm ${darkMode ? "text-white" : "text-gray-900"}`}>
                    x{speeds[speedIndex]}
                    </span>               
                </div>

                <div className="flex flex-col items-center gap-2 mt-2">
                    <div className="flex items-center gap-1">
                        <span className="text-gray-600">⏱</span>
                        <span className={`font-mono text-sm ${darkMode ? "text-white" : "text-gray-900"}`}>
                            {formatTime(currentSimulTime)}
                        </span>
                    </div>
                </div>
                </div>
            </div>
            </main>

            <footer ref={footerRef} className="flex flex-col md:flex-row items-center justify-between p-4 bg-[#E2E8F0] shadow-lg footer">
            <div className="flex flex-col md:flex-row gap-6 items-center">
                <div className="flex flex-col items-center gap-4">
                <span className="text-sm font-semibold">OHT Mode</span>

                <label className="flex items-center cursor-pointer">
                    <input type="radio" name="ohtMode" value="random" checked={ohtMode === "random"} onChange={() => setOhtMode("random")} className="hidden" />
                    <span
                    className={`px-3 py-1 rounded-lg transition text-sm font-medium cursor-pointer ${
                        ohtMode === "random" ? "bg-blue-600 text-white" : "bg-gray-300 text-black"
                    }`}
                    >
                    Random
                    </span>
                </label>

                <label className="flex items-center cursor-pointer">
                    <input type="radio" name="ohtMode" value="file" checked={ohtMode === "file"} onChange={() => setOhtMode("file")} className="hidden" />
                    <span
                    className={`px-3 py-1 rounded-lg transition text-sm font-medium cursor-pointer ${
                        ohtMode === "file" ? "bg-blue-600 text-white" : "bg-gray-300 text-black"
                    }`}
                    >
                    File Upload
                    </span>
                </label>
                </div>

                <div className="flex items-center gap-3">
                <input
                    type="checkbox"
                    checked={isAccelEnabled}
                    onChange={(e) => setIsAccelEnabled(e.target.checked)}
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
                    className="border border-gray-400 bg-white text-black"
                    />
                </div>
                )}

                <div className="flex flex-col items-center gap-3">
                <label className="text-sm font-semibold">Max Time</label>
                <TimeInput
                    ref={maxTimeref}
                    value={maxTime}
                    onChange={setMaxTime}
                    className="border border-gray-400 bg-white text-black"
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
                    className="p-2 rounded-md border border-gray-400 bg-white text-black focus:outline-none focus:ring focus:ring-blue-500 w-32 text-center"
                    />
                </div>
                )}
            </div>

            {ohtMode === "file" && (
                <div className="flex flex-col md:flex-row gap-6 items-center mt-4">
                <div className="flex flex-col items-center">
                    <label className="flex flex-col items-center px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg shadow-md transition cursor-pointer">
                    📂 Upload Job File
                    <input ref={jobFileInputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
                    </label>
                    {selectedJobFile ? (
                    <p className="text-sm text-green-600 mt-2">{(selectedJobFile as any).name}</p>
                    ) : (
                    <p className="text-sm text-gray-500 mt-2">No file selected</p>
                    )}
                </div>

                <div className="flex flex-col items-center">
                    <label className="flex flex-col items-center px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg shadow-md transition cursor-pointer">
                    📂 Upload OHT File
                    <input ref={OhtFileInputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
                    </label>
                    {selectedOhtFile ? (
                    <p className="text-sm text-green-600 mt-2">{(selectedOhtFile as any).name}</p>
                    ) : (
                    <p className="text-sm text-gray-500 mt-2">No file selected</p>
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
                    if (!isRunningBack) startBackSimulation();
                    else stopBackSimulation();
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
