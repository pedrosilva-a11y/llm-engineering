import { AppHeader, CompletionWorkspace, ConfigurationSelector } from './components'

import './App.css'

function App() {
  return (
    <>
      <AppHeader />

      <main className="app">
        <CompletionWorkspace />
        <ConfigurationSelector />
      </main>
    </>
  )
}

export default App
