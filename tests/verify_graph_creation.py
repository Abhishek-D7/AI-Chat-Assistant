
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.langgraph_graph import get_agent_graph

async def main():
    print("🔨 Testing Graph Creation...")
    try:
        graph = get_agent_graph()
        print("✅ Graph created successfully!")
        
        # Optional: Print graph structure if possible
        # print(graph.get_graph().draw_ascii())
        
    except Exception as e:
        print(f"❌ Graph creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
