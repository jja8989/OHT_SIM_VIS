"use client"
import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { getClientId } from './utils/getClientId';


const client_id = getClientId();

const OHTVisualization = dynamic(() => import('./components/OHTVisualization'), { ssr: false });

interface Node {
    id: string;
    x: number;
    y: number;
}

interface Rail {
    from: string;
    to: string;
}

interface Port {
    name: string;
    x: number;
    y: number;
    rail_line: string;
}

interface LayoutData {
    nodes: Node[];
    rails: Rail[];
    ports: Port[];
}

const socket = io('/', {
    path: '/socket.io',
    transports: ['websocket'],
    query: {
        client_id: client_id,
      }
  });


export default function Home() {
    const [data, setData] = useState<LayoutData | null>(null);

    useEffect(() => {
        socket.emit('layout');

        socket.on('layoutData', (layoutData: LayoutData) => {
            setData(layoutData); 
        });


        return () => {
            socket.off('layoutData');
        };
    }, []);

    return (
        <div>
            <main>
                {data && <OHTVisualization data={data} />}
            </main>
        </div>
    );
}
