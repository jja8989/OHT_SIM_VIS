"use client"
// pages/page.tsx
import Head from 'next/head';
import dynamic from 'next/dynamic';
import { useEffect, useState } from 'react';

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

export default function Home() {
    const [data, setData] = useState<LayoutData | null>(null);

    useEffect(() => {
        fetch('http://localhost:5001/layout')
            .then(response => response.json())
            .then(setData);
    }, []);

    return (
        <div>
            <main>
                {data && <OHTVisualization data={data} />}
            </main>
        </div>
    );
}
