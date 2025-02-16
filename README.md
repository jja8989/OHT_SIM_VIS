# OHT Simulation & Visualization System

This project is an **OHT (Overhead Hoist Transport) simulation and visualization system**, designed to **simulate, visualize, and analyze** OHT movements in a factory or semiconductor manufacturing environment. The system ensures **real-time pathfinding, collision avoidance, and dynamic updates** using a **Flask backend** and a **Next.js frontend** with WebSockets.

---

## ** Features**
- **Real-time OHT simulation** with dynamic movements and collision detection  
- **Graph-based pathfinding** using **NetworKit**  
- **WebSocket-based real-time updates** between **Flask (backend)** and **Next.js (frontend)**  
- **Interactive visualization** using **d3.js**  
- **Docker-based deployment** for easy setup and execution  

---

## **Installation & Setup**
### **1️⃣ Prerequisites**
Before starting, make sure you have **Docker** and **Docker Compose** installed.

### **2️⃣ Clone the Repository**
```sh
git clone <repository-url>
cd <project-directory>
```

### **3️⃣ Start the System**

Run the following command to start both the backend (Flask) and frontend (Next.js):

```sh
docker-compose up -d
```
This will launch all required services in detached mode.

To stop the services, use:
```sh
docker-compose down
```