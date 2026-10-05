import { AppHeader, CompletionPanel, ConfigurationSelector } from './components'

import './App.css'

function App() {
  return (
    <>
      <AppHeader />

      <main className="app">
        <ConfigurationSelector />
        <CompletionPanel />
      </main>
    </>
  )
}

export default App
