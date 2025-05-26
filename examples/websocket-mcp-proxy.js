#!/usr/bin/env node

/**
 * WebSocket MCP Proxy for Claude Code
 * 
 * This proxy allows Claude Code to connect to a remote MCP server over WebSocket.
 * It translates between Claude Code's local MCP protocol and the remote WebSocket connection.
 */

const WebSocket = require('ws');
const readline = require('readline');
const { EventEmitter } = require('events');

class MCPWebSocketProxy extends EventEmitter {
  constructor(wsUrl, apiKey) {
    super();
    this.wsUrl = wsUrl;
    this.apiKey = apiKey;
    this.ws = null;
    this.messageQueue = [];
    this.connected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  async connect() {
    try {
      this.ws = new WebSocket(this.wsUrl, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'X-Client-Type': 'claude-code-proxy',
          'X-Client-Version': '1.0.0'
        },
        // Handle self-signed certificates in development
        rejectUnauthorized: process.env.NODE_ENV === 'production'
      });

      this.ws.on('open', () => {
        console.error('Connected to remote MCP server');
        this.connected = true;
        this.reconnectAttempts = 0;
        
        // Send any queued messages
        while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          this.ws.send(message);
        }
        
        // Send initialization message
        this.send({
          jsonrpc: '2.0',
          method: 'initialize',
          params: {
            protocolVersion: '2024-11-05',
            capabilities: {
              roots: {},
              sampling: {}
            },
            clientInfo: {
              name: 'claude-code-proxy',
              version: '1.0.0'
            }
          },
          id: 1
        });
      });

      this.ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          // Forward to stdout for Claude Code
          console.log(JSON.stringify(message));
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      });

      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error.message);
      });

      this.ws.on('close', (code, reason) => {
        console.error(`WebSocket closed: ${code} - ${reason}`);
        this.connected = false;
        
        // Attempt to reconnect
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          console.error(`Reconnecting in ${this.reconnectDelay}ms... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          setTimeout(() => this.connect(), this.reconnectDelay);
          this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Max 30 seconds
        } else {
          console.error('Max reconnection attempts reached. Exiting.');
          process.exit(1);
        }
      });

      this.ws.on('ping', () => {
        this.ws.pong();
      });

    } catch (error) {
      console.error('Connection error:', error);
      process.exit(1);
    }
  }

  send(message) {
    const data = JSON.stringify(message);
    if (this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      // Queue message if not connected
      this.messageQueue.push(data);
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Main execution
async function main() {
  // Get configuration from command line or environment
  const wsUrl = process.argv[2] || process.env.MCP_WS_URL;
  const apiKey = process.env.MCP_API_KEY;

  if (!wsUrl) {
    console.error('Error: WebSocket URL required');
    console.error('Usage: node websocket-mcp-proxy.js <websocket-url>');
    console.error('Or set MCP_WS_URL environment variable');
    process.exit(1);
  }

  if (!apiKey) {
    console.error('Error: MCP_API_KEY environment variable required');
    process.exit(1);
  }

  // Create proxy instance
  const proxy = new MCPWebSocketProxy(wsUrl, apiKey);
  
  // Set up stdin reader for messages from Claude Code
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  rl.on('line', (line) => {
    try {
      const message = JSON.parse(line);
      proxy.send(message);
    } catch (error) {
      console.error('Error parsing input:', error);
    }
  });

  // Handle process termination
  process.on('SIGINT', () => {
    console.error('Shutting down...');
    proxy.close();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.error('Shutting down...');
    proxy.close();
    process.exit(0);
  });

  // Connect to remote server
  await proxy.connect();
}

// Error handling
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Start the proxy
main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});