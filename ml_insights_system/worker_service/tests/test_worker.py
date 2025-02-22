import asyncio
import json
import aio_pika

async def send_test_message():
    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust("amqp://guest:guest@localhost:5672/")
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Prepare the message
        test_message = {
            "sentence_id": "test-123",
            "priority": "high"
        }
        
        # Create a message
        message = aio_pika.Message(
            body=json.dumps(test_message).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        # Send the message
        await channel.default_exchange.publish(
            message,
            routing_key="sentence_processing"
        )
        
        print(f"Sent test message: {test_message}")

if __name__ == "__main__":
    asyncio.run(send_test_message())