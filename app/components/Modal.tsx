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
    const [simulationData, setSimulationData] = useState<any[]>([]);

    useEffect(() => {
        socket.emit('get_simulation_tables');

        socket.on('simulation_tables', (data) => {
            const sortedTables = data.tables.sort((a, b) => {
                const numA = parseInt(a.replace("simulation_", ""), 10);
                const numB = parseInt(b.replace("simulation_", ""), 10);
                return numB - numA;  // ✅ 숫자가 큰(최신) 테이블이 위로
            });

            setSimulationTables(sortedTables);
        });

        return () => {
            socket.off('simulation_tables');
        };
    }, []);

    const fetchSimulationData = (tableName: string) => {
        setSelectedTable(tableName);

            
        socket.off("simulation_data");

        socket.emit("get_simulation_data", { table_name: tableName });

        socket.on("simulation_data", (data) => {
        
            const pivotMap: Record<string, Record<string, number>> = {};
            const allEdges = new Set<string>();

            for (const { time, edge_id, avg_speed } of data.data) {
            if (!pivotMap[time]) pivotMap[time] = { time };
            // (선택) 여기서도 방어적으로 숫자화
            const v = typeof avg_speed === "number" ? avg_speed : Number(avg_speed);
            pivotMap[time][edge_id] = v;
            allEdges.add(edge_id);
            }

            const times = Object.keys(pivotMap).sort();

            const edgeHeaders = Array.from(allEdges).sort();
            const rows: any[] = [];

            for (const t of times) {
            const row = { time: t } as Record<string, any>;
            for (const e of edgeHeaders) {
                // 없는 값은 빈칸 또는 null로
                row[e] = pivotMap[t][e] ?? "";
            }
            rows.push(row);
            }

            setSimulationData(rows);
        
        });
    };
    
    const downloadCSV = () => {
        if (!selectedTable || simulationData.length === 0) return;

        const headerSet = new Set<string>(["time"]);
        for (const r of simulationData) {
            Object.keys(r).forEach((k) => headerSet.add(k));
        }
        headerSet.delete("time");
        const headers = ["time", ...Array.from(headerSet).sort()];

        const esc = (v: any) => {
            if (v === null || v === undefined) return "";
            const s = String(v);
            return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        };

        const lines = [
            headers.join(","),
            ...simulationData.map((row) => headers.map((h) => esc(row[h])).join(",")),
        ];

        const csvContent = "data:text/csv;charset=utf-8," + lines.join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        const timestamp = new Date().toISOString().replace(/[-:]/g, "").split(".")[0];
        link.setAttribute("download", `${selectedTable}_${timestamp}.csv`);
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
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg w-1/2">
                <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Simulation Results</h2>
                

                <div className="mb-4">
                    <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Available Simulations</h3>

                    <ul className="mt-2 max-h-40 overflow-y-auto border border-gray-300 dark:border-gray-600 rounded p-2">
                        {simulationTables.length > 0 ? (
                            simulationTables.map((table) => (
                                <li 
                                    key={table} 
                                    className="flex justify-between items-center cursor-pointer p-2 hover:bg-gray-200 dark:hover:bg-gray-700"
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
                            <p className="text-gray-500 dark:text-gray-400">No simulations found.</p>
                        )}
                    </ul>
                </div>

                {selectedTable && (
                    <div>
                        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">Simulation Data: {selectedTable}</h3>
                        <div className="max-h-60 overflow-y-auto mt-2 border border-gray-300 dark:border-gray-600 rounded p-2">
                            <pre className="text-xs text-gray-800 dark:text-gray-200">{JSON.stringify(simulationData, null, 2)}</pre>
                        </div>

                        <button 
                            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            onClick={downloadCSV}
                        >
                            Download CSV
                        </button>
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
