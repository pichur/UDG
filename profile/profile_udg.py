import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from line_profiler import LineProfiler, show_text
import discrete_disk
import udg
import Graph6Converter

sys.stdout.reconfigure(encoding='utf-8')

def run_a():
    """D^{ 5_K_5_m"""
    nxg = Graph6Converter.edge_list_to_graph('5: 1,3 ; 1,4 ; 1,5 ; 2,3 ; 2,4 ; 2,5 ; 3,4 ; 3,5 ; 4,5')
    g = udg.Graph(nxg)
    g.set_unit(8)
    g.udg_recognition()

def run_b():
    nxg = Graph6Converter.edge_list_to_graph('9: 1,2 ; 1,3 ; 1,4 ; 1,5 ; 5,6 ; 5,7 ; 5,8 ; 8,9 ; 8,1')
    g = udg.Graph(nxg)
    g.set_unit(16)
    g.udg_recognition()

def print_stats_over(lp: LineProfiler, limit: float = 0.1, *, stream=None):
    """
    Wypisz wynik line-profilera pomijając funkcje,
    których łączny czas < limit (sekundy).
    """
    lstats = lp.get_stats()                     # LineStats
    unit   = lstats.unit                        # sekunda / tick

    # Przefiltruj dict {(fn, lineno, name): [(lineno, nhits, ticks)…]}
    timings = {
        key: lines
        for key, lines in lstats.timings.items()
        if sum(t[2] for t in lines) * unit >= limit
    }

    # Jeśli nic nie zostało, poinformuj użytkownika
    if not timings:
        print(f"No functions ≥ {limit:.3f} s", file=stream or sys.stdout)
        return

    print(f"Show only functions ≥ {limit:.3f} s", file=stream or sys.stdout)

    # show_text() to ta sama funkcja, której używa print_stats()
    show_text(
        timings,
        unit,
        output_unit=None,          # domyślnie µs
        stream=stream,
        stripzeros=False,          # filtrujemy sami
    )


if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_module(discrete_disk)
    lp.add_module(udg)
    lp(run_b)()
    print_stats_over(lp, limit=0.1)
