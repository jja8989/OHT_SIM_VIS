"use client"
// pages/page.tsx
import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { getClientId } from './utils/getClientId';


const client_id = getClientId();

// console.log(client_id);


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

// const socket = io('http://localhost:5000'); // 소켓 서버 주소

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
        // fetch('http://localhost:5001/layout')
        //     .then(response => response.json())
        //     .then(setData);

                // 서버로 layout 데이터를 요청
        socket.emit('layout'); // `requestLayout` 이벤트를 서버로 보냄

        // 서버로부터 layout 데이터를 수신
        socket.on('layoutData', (layoutData: LayoutData) => {
            setData(layoutData); // 데이터를 상태에 저장
        });

        // 클린업: 컴포넌트 언마운트 시 소켓 이벤트 제거
        return () => {
            socket.off('layoutData'); // 이벤트 리스너 제거
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
