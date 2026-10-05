import { AppHeader } from './components/AppHeader'
import { CompletionPanel } from './components/CompletionPanel'

import './App.css'

function App() {
  return (
    <>
      <AppHeader />

      <main className="app">
        <CompletionPanel />
      </main>
    </>
  )
}

export default App
