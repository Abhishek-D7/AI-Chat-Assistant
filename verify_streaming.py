import asyncio
from app.langgraph_graph import process_user_message_with_context_streaming

async def run():
    print('--- Turn 1 ---')
    async for x in process_user_message_with_context_streaming('My name is Abhi', 'u1', 'Abhi', thread_id='t_stream_1'):
        if x['type'] == 'token':
            print(x['content'], end='', flush=True)
    print('\n')

    print('--- Turn 2 ---')
    async for x in process_user_message_with_context_streaming('What is my name?', 'u1', 'Abhi', thread_id='t_stream_1'):
        if x['type'] == 'token':
            print(x['content'], end='', flush=True)
    print('\n')

if __name__ == "__main__":
    asyncio.run(run())
