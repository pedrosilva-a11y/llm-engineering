import { AppHeader, CompletionPanel, ConfigurationSelector } from './components'

import './App.css'

function App() {
  return (
    <>
      <AppHeader />

      <main className="app">
        <CompletionPanel />
        <ConfigurationSelector />
      </main>
    </>
  )
}

export default App
