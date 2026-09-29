import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // The API only listens on 127.0.0.1; xfwd appends each visitor's real
      // address, which is how the API tells the presenter's machine from
      // guests on the network. `vite preview` inherits this proxy.
      '/api': { target: 'http://127.0.0.1:8000', xfwd: true },
    },
  },
})
