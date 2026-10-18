// PROTOTYPE — throwaway. Five variants of the main screen, switchable via ?variant=A..E and the bottom bar.
// Fake model, fake field numbers; the map (OpenFreeMap liberty + NASA GIBS tiles) is real.
import { createRoot } from 'react-dom/client'
import './styles.css'
import VariantA from './VariantA.jsx'
import VariantB from './VariantB.jsx'
import VariantC from './VariantC.jsx'
import VariantD from './VariantD.jsx'
import VariantE from './VariantE.jsx'
import { Switcher } from './shared.jsx'

const names = { A: 'Workbench', B: 'Map-first', C: 'Chat-first', D: 'Map + chat', E: 'Focus' }
const variant = new URLSearchParams(location.search).get('variant') ?? 'A'

createRoot(document.getElementById('root')).render(
  <>
    {variant === 'A' && <VariantA />}
    {variant === 'B' && <VariantB />}
    {variant === 'C' && <VariantC />}
    {variant === 'D' && <VariantD />}
    {variant === 'E' && <VariantE />}
    <Switcher variants={['A', 'B', 'C', 'D', 'E']} current={variant} names={names} />
  </>
)
