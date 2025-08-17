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
        socket.emit("get_simulation_data", { table_name: tableName });
    
        socket.on("simulation_data", (data) => {

            const pivotData: Record<string, Record<string, number>> = {};
    
            data.data.forEach(({ time, edge_id, avg_speed }) => {
                if (!pivotData[time]) pivotData[time] = { time };
                pivotData[time][edge_id] = avg_speed;
            });
    
            setSimulationData(Object.values(pivotData)); 
        });
    };
    

    const downloadCSV = () => {
        if (!selectedTable || simulationData.length === 0) return;

        const headers = Object.keys(simulationData[0]).join(",");
        const rows = simulationData.map((row) =>
            Object.values(row).join(",")
        ).join("\n");
    
        const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;
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
