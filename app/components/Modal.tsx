import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { getClientId } from '../utils/getClientId';





const client_id = getClientId();


const socket = io(process.env.NEXT_PUBLIC_SOCKET_URL || '/', {
    path: '/socket.io',
    transports: ['websocket'],
    query: {
        client_id: client_id,
      }
  });

interface ModalProps {
    onClose: () => void;
}

const Modal: React.FC<ModalProps> = ({ onClose }) => {
    const [simulationTables, setSimulationTables] = useState<string[]>([]);
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [simulationData, setSimulationData] = useState<{
            avg: any[];
            count: any[];
            combined: any[];
            }>({ avg: [], count: [], combined: [] });

    const [viewMode, setViewMode] = useState<"avg" | "count">("avg");

    const [loading, setLoading] = useState(false);


    useEffect(() => {
        socket.emit('get_simulation_tables');

        socket.on('simulation_tables', (data) => {
            const sortedTables = data.tables.sort((a, b) => {
                const numA = parseInt(a.replace("simulation_", ""), 10);
                const numB = parseInt(b.replace("simulation_", ""), 10);
                return numB - numA;  
            });

            setSimulationTables(sortedTables);
        });

        return () => {
            socket.off('simulation_tables');
        };
    }, []);

    const fetchSimulationData = (tableName: string) => {

        setLoading(true);


        setSelectedTable(tableName);

            
        socket.off("simulation_data");

        socket.emit("get_simulation_data", { table_name: tableName });

        socket.on("simulation_data", (data) => {
        
            const pivotMapAvg: Record<string, Record<string, number | string>> = {};
            const pivotMapCount: Record<string, Record<string, number | string>> = {};
            const allEdges = new Set<string>();

            for (const { time, edge_id, avg_speed, count } of data.data) {
                if (!pivotMapAvg[time]) pivotMapAvg[time] = { time };
                if (!pivotMapCount[time]) pivotMapCount[time] = { time };

                const v = typeof avg_speed === "number" ? avg_speed : Number(avg_speed);
                const c = typeof count === "number" ? count : Number(count);

                pivotMapAvg[time][edge_id] = v;
                pivotMapCount[time][edge_id] = c;

                allEdges.add(edge_id);
            }

            const times = Object.keys(pivotMapAvg).sort();
            const edgeHeaders = Array.from(allEdges).sort();

            const rowsAvg: any[] = [];
            const rowsCount: any[] = [];
            const rowsCombined: any[] = [];

            for (const t of times) {
                const rowAvg: Record<string, any> = { time: t };
                const rowCount: Record<string, any> = { time: t };
                const rowCombined: Record<string, any> = { time: t };

                for (const e of edgeHeaders) {
                    rowAvg[e] = pivotMapAvg[t][e] ?? "";
                    rowCount[e] = pivotMapCount[t][e] ?? "";
                    rowCombined[`${e}_avg_speed`] = pivotMapAvg[t][e] ?? "";
                    rowCombined[`${e}_count`] = pivotMapCount[t][e] ?? "";
                }

                rowsAvg.push(rowAvg);
                rowsCount.push(rowCount);
                rowsCombined.push(rowCombined);
            }

            setSimulationData({
                avg: rowsAvg,
                count: rowsCount,
                combined: rowsCombined
            });

            setLoading(false);


            // const times = Object.keys(pivotMap).sort();

            // const edgeHeaders = Array.from(allEdges).sort();
            // const rows: any[] = [];

            // for (const t of times) {
            // const row = { time: t } as Record<string, any>;
            // for (const e of edgeHeaders) {
            //     // 없는 값은 빈칸 또는 null로
            //     row[e] = pivotMap[t][e] ?? "";
            // }
            // rows.push(row);
            // }

            // setSimulationData(rows);
        
        });
    };
    
    // const downloadCSV = () => {
    //     if (!selectedTable || simulationData.length === 0) return;

    //     const headerSet = new Set<string>(["time"]);
    //     for (const r of simulationData) {
    //         Object.keys(r).forEach((k) => headerSet.add(k));
    //     }
    //     headerSet.delete("time");
    //     const headers = ["time", ...Array.from(headerSet).sort()];

    //     const esc = (v: any) => {
    //         if (v === null || v === undefined) return "";
    //         const s = String(v);
    //         return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    //     };

    //     const lines = [
    //         headers.join(","),
    //         ...simulationData.map((row) => headers.map((h) => esc(row[h])).join(",")),
    //     ];

    //     const csvContent = "data:text/csv;charset=utf-8," + lines.join("\n");
    //     const encodedUri = encodeURI(csvContent);
    //     const link = document.createElement("a");
    //     link.setAttribute("href", encodedUri);
    //     const timestamp = new Date().toISOString().replace(/[-:]/g, "").split(".")[0];
    //     link.setAttribute("download", `${selectedTable}_${timestamp}.csv`);
    //     document.body.appendChild(link);
    //     link.click();
    //     };

    const downloadCSV = (rows: any[], filename: string) => {
        if (!selectedTable || rows.length === 0) return;

        const headerSet = new Set<string>(["time"]);
        for (const r of rows) Object.keys(r).forEach((k) => headerSet.add(k));
        headerSet.delete("time");
        const headers = ["time", ...Array.from(headerSet).sort()];

        const esc = (v: any) => {
            if (v === null || v === undefined) return "";
            const s = String(v);
            return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        };

        const lines = [
            headers.join(","),
            ...rows.map((row) => headers.map((h) => esc(row[h])).join(","))
        ];

        const csvContent = "data:text/csv;charset=utf-8," + lines.join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        const timestamp = new Date().toISOString().replace(/[-:]/g, "").split(".")[0];
        link.setAttribute("download", `${selectedTable}_${filename}_${timestamp}.csv`);
        document.body.appendChild(link);
        link.click();
    };


    const deleteSimulationTable = (tableName: string) => {
        if (!window.confirm(`Are you sure you want to delete ${tableName}?`)) return;
    
        socket.emit("delete_simulation_table", { table_name: tableName });
    
        socket.on("table_deleted", (data) => {
            alert(data.message);
            if (data.success) {
                setSimulationTables((prevTables) => prevTables.filter((table) => table !== tableName));
            }
        });
    };

    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div className="bg-white p-6 rounded-lg shadow-lg w-1/2">
                <h2 className="text-lg font-semibold mb-4 text-gray-900">Simulation Results</h2>
                

                <div className="mb-4">
                    <h3 className="text-sm font-medium text-gray-700">Available Simulations</h3>

                    <ul className="mt-2 max-h-40 overflow-y-auto border border-gray-300 rounded p-2">
                        {simulationTables.length > 0 ? (
                            simulationTables.map((table) => (
                                <li 
                                    key={table} 
                                    className="flex justify-between items-center cursor-pointer p-2 hover:bg-gray-200"
                                >
                                    <span onClick={() => fetchSimulationData(table)} className="flex-grow">
                                        {table}
                                    </span>

                                    <button 
                                        className="text-red-500 hover:underline ml-4"
                                        onClick={() => deleteSimulationTable(table)}
                                    >
                                        ❌
                                    </button>
                                </li>
                            ))
                        ) : (
                            <p className="text-gray-500">No simulations found.</p>
                        )}
                    </ul>
                </div>

                {selectedTable && (
                    <div>
                        <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium text-gray-700">
                            Simulation Data: {selectedTable}
                        </h3>

      
                        <label className="flex items-center cursor-pointer space-x-2">
                        <span className="text-xs font-medium text-gray-700">Avg Speed</span>
                        <div className="relative">
                            <input
                            type="checkbox"
                            className="sr-only"
                            checked={viewMode === "count"}
                            onChange={() => setViewMode(viewMode === "avg" ? "count" : "avg")}
                            />
                            <div className="block w-12 h-6 rounded-full border border-gray-400"></div>
                            <div
                            className={`dot absolute left-1 top-1 w-4 h-4 rounded-full bg-gray-600 transition ${
                                viewMode === "count" ? "translate-x-6" : ""
                            }`}
                            ></div>
                        </div>
                        <span className="text-xs font-medium text-gray-700">Count</span>
                        </label>

                        </div>


                        {simulationData[viewMode] && simulationData[viewMode].length > 0 ? (
                        <div className="relative max-h-60 overflow-y-auto mt-2 border border-gray-300 rounded">
                            <div className={loading ? "blur-sm pointer-events-none" : ""}>

                            <table className="w-full text-xs border-collapse">
                            <thead>
                                <tr>
                                {["time", ...Object.keys(simulationData[viewMode][0]).filter((k) => k !== "time").slice(0, 5)].map(
                                    (h) => (
                                    <th
                                        key={h}
                                        className="px-2 py-1 text-[10px] font-medium text-gray-700 whitespace-nowrap"
                                    >
                                        {h}
                                    </th>
                                    )
                                )}
                                </tr>
                            </thead>
                            <tbody>
                                {simulationData[viewMode].slice(0, 20).map((row, i) => {
                                const edgeCols = Object.keys(row).filter((k) => k !== "time").slice(0, 5);
                                const headers = ["time", ...edgeCols];
                                return (
                                    <tr key={i} className="hover:bg-gray-50">
                                    {headers.map((h) => (
                                        <td
                                        key={h}
                                        className="px-2 py-1 border border-gray-300 text-gray-700"
                                        >
                                        {row[h]}
                                        </td>
                                    ))}
                                    </tr>
                                );
                                })}
                            </tbody>
                            </table>
                            </div>

                            {loading && (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-blue-500"></div>
                                <span className="ml-2 text-gray-700 dark:text-gray-200 text-sm">Loading...</span>
                            </div>
                            )}
                        </div>

                        ) : (
                        <p className="text-sm text-gray-500">No data available</p>
                        )}

                        {simulationData[viewMode].length > 0 &&
                            <p className="text-xs text-gray-500 mt-1">
                                Showing first 20 rows & first 5 edges
                            </p>
                        }



                        <div className="mt-4 flex gap-2 text-sm">
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={() => downloadCSV(simulationData.avg, "avg")}
                        >
                            Download Avg Speed
                        </button>
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={() => downloadCSV(simulationData.count, "count")}
                        >
                            Download Count
                        </button>
                        <button
                            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={() => downloadCSV(simulationData.combined, "combined")}
                        >
                            Download Combined
                        </button>
                        </div>
                    </div>
                    )}


                <div className="mt-4 text-right">
                    <button className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Modal;
