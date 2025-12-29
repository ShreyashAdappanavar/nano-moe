# from graphviz import Digraph

# def draw_configured_architecture():
#     # Configuration strings
#     dims_txt = "D_Model: 192\nVocab: 10k"
#     attn_txt = "Heads: 8\nKV Heads: 2\nRoPE Theta: 10k"
#     cache_txt = "Max Seq: 512\nBatch: 64"
#     moe_txt = "Total Experts: 4\nTop-K: 2\nHidden Dim: 512"
#     shared_txt = "Shared Experts: 1"
    
#     dot = Digraph(comment='Configured Transformer Architecture', format='png')
#     dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')
#     dot.attr('node', fontname='Helvetica', fontsize='12')

#     # --- Main Input ---
#     dot.node('In', f'Input Embedding\n({dims_txt})', shape='invhouse', style='filled', fillcolor='#E0E0E0')

#     # --- Transformer Block ---
#     with dot.subgraph(name='cluster_block') as b:
#         b.attr(label='Transformer Block (x6 Layers)', style='dashed', color='#555555', bgcolor='#FAFAFA', fontname='Helvetica-Bold')

#         # 1. Attention Cluster
#         b.node('LN1', 'RMSNorm\n(eps: 1e-06)', style='filled', fillcolor='white', fontsize='10')
        
#         with b.subgraph(name='cluster_attn') as a:
#             a.attr(label=f'Causal Self-Attention\n({attn_txt})', color='#7E57C2', style='rounded', bgcolor='#EDE7F6')
            
#             # Projections
#             a.node('QKV', 'Linear Projections\nQ: (B, T, 8, 24)\nKV: (B, T, 2, 24)', shape='box', fillcolor='white', fontsize='11')
            
#             # RoPE
#             with a.subgraph(name='cluster_rope') as r:
#                 r.attr(label='Rotary Positional Embeddings', color='#5E35B1', style='dashed')
#                 r.node('RoPE_Op', 'Rotate Q & K', shape='parallelogram', fillcolor='#D1C4E9')

#             # KV Cache
#             with a.subgraph(name='cluster_cache') as c:
#                 c.attr(label='Inference Cache', color='#D84315', style='bold')
#                 c.node('KVCache', f'KV Cache\n({cache_txt})', shape='cylinder', style='filled', fillcolor='#FFCCBC')
#                 c.node('Concat', 'Concat', shape='point')

#             # Attention Mech
#             a.node('AttnScore', 'Scaled Dot-Product\n(Grouped Query Attn)', shape='box', fillcolor='white')
#             a.node('OProj', 'Output Projection\n(192 -> 192)', shape='box', fillcolor='white')
            
#             # Edges inside Attention
#             a.edge('QKV', 'RoPE_Op', label='Q, K')
#             a.edge('QKV', 'KVCache', label='V', style='dashed')
#             a.edge('RoPE_Op', 'KVCache', label='Rotated K')
            
#             a.edge('RoPE_Op', 'AttnScore', label='Q')
#             a.edge('KVCache', 'AttnScore', label='History K,V')
#             a.edge('AttnScore', 'OProj')

#         b.node('Add1', '+', shape='circle', width='0.3', fixedsize='true')
#         b.node('LN2', 'RMSNorm', style='filled', fillcolor='white', fontsize='10')

#         # 2. MoE Cluster
#         with b.subgraph(name='cluster_moe') as m:
#             m.attr(label=f'DeepSeek-Style MoE\n({moe_txt})', color='#1E88E5', style='rounded', bgcolor='#E3F2FD')
            
#             m.node('Router', 'Router\n(192 -> 4)', shape='hexagon', style='filled', fillcolor='#FFAB91')
#             m.node('Shared', f'{shared_txt}\n(Always Active)', shape='box', style='filled', fillcolor='#C8E6C9')
            
#             with m.subgraph(name='cluster_routed_ex') as re:
#                 re.attr(label='Routed Experts', style='invis')
#                 re.node('Experts', 'Select Top-2 Experts\n(FFN: 192 -> 512 -> 192)', shape='box', style='filled', fillcolor='#FFF9C4')
            
#             m.node('MoEAgg', 'Weighted Sum', shape='diamond', style='filled', fillcolor='white')
            
#             # Edges inside MoE
#             m.edge('Router', 'Experts', label='Softmax Weights', color='#E65100', fontsize='10')
#             m.edge('Experts', 'MoEAgg')
#             m.edge('Shared', 'MoEAgg')

#         b.node('Add2', '+', shape='circle', width='0.3', fixedsize='true')

#         # Connect Block Components
#         b.edge('LN1', 'QKV')
#         b.edge('OProj', 'Add1')
#         b.edge('Add1', 'LN2')
#         b.edge('LN2', 'Router')
#         b.edge('LN2', 'Shared')
#         b.edge('MoEAgg', 'Add2')

#     # Residuals
#     dot.edge('In', 'Add1', style='dotted', label='residual')
#     dot.edge('Add1', 'Add2', style='dotted', label='residual')

#     # Output
#     dot.node('Out', 'Logits\n(vocab: 10000)', shape='oval', style='filled', fillcolor='#E0E0E0')
#     dot.edge('Add2', 'Out')
#     dot.edge('In', 'LN1')

#     dot.render('configured_architecture_viz', view=False)
#     print("Diagram generated as 'configured_architecture_viz.png'")

# if __name__ == "__main__":
#     draw_configured_architecture()

from graphviz import Digraph

def draw_aesthetic_architecture():
    # --- Style Configuration ---
    # Palette: Modern Slate & Indigo
    colors = {
        'bg': '#ffffff',
        'block_bg': '#f8f9fa',
        'block_border': '#dee2e6',
        'input': '#e9ecef',
        'attn_header': '#e8eaf6',   # Indigo 50
        'attn_fill': '#ffffff',
        'attn_border': '#c5cae9',   # Indigo 100
        'moe_header': '#e0f2f1',    # Teal 50
        'moe_fill': '#ffffff',
        'moe_border': '#b2dfdb',    # Teal 100
        'router': '#ffe0b2',        # Orange 100
        'edge': '#546e7a',
        'text_main': '#263238',
        'text_dim': '#607d8b'
    }
    
    dot = Digraph(comment='Aesthetic Transformer', format='png')
    
    # Global Graph Attributes for "Pretty" look
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6')
    dot.attr(bgcolor=colors['bg'])
    dot.attr('node', shape='plain', fontname='Helvetica', fontsize='11')
    dot.attr('edge', color=colors['edge'], penwidth='1.2', arrowsize='0.7')

    # --- HTML Label Helpers ---
    def make_card(title, subtitle, details, color_header, color_border, width=200):
        # Creates a HTML-like table node that looks like a UI card
        rows = ""
        for k, v in details.items():
            rows += f'<tr><td align="left" cellpadding="2"><font color="{colors["text_dim"]}" point-size="10">{k}</font></td><td align="right" cellpadding="2"><font color="{colors["text_main"]}" point-size="10"><b>{v}</b></font></td></tr>'
        
        return f'''<
            <table border="0" cellborder="1" cellspacing="0" cellpadding="0" width="{width}" color="{color_border}" bgcolor="white" style="rounded">
                <tr><td colspan="2" bgcolor="{color_header}" cellpadding="5" border="1" sides="b">
                    <font color="{colors["text_main"]}" point-size="12"><b>{title}</b></font><br/>
                    <font color="{colors["text_dim"]}" point-size="9">{subtitle}</font>
                </td></tr>
                {rows}
            </table>
        >'''

    # --- 1. Inputs ---
    dot.node('In', make_card(
        "Input Embedding", 
        "Tokens → Vectors", 
        {"Batch": "64", "Seq Len": "512", "Dim": "192", "Vocab": "10k"}, 
        colors['input'], colors['block_border']
    ))

    # --- 2. Transformer Block Container ---
    with dot.subgraph(name='cluster_block') as b:
        b.attr(label='Transformer Layer (Stack x6)', fontname='Helvetica-Bold', fontsize='14', color=colors['block_border'], bgcolor=colors['block_bg'], style='rounded')
        
        # Pre-Attn Norm
        b.node('LN1', 'RMSNorm', shape='box', style='rounded,filled', fillcolor='white', color=colors['edge'], width='1.5')
        
        # Attention Block (Detailed Card)
        attn_details = {
            "Heads (Q)": "8",
            "KV Heads": "2 (GQA)",
            "Head Dim": "24",
            "RoPE": "Rotary",
            "Cache": "Active"
        }
        b.node('Attn', make_card("Causal Self-Attention", "Masked + RoPE + KV Cache", attn_details, colors['attn_header'], colors['attn_border']))
        
        # Residual 1
        b.node('Add1', '+', shape='circle', style='filled', fillcolor='white', color=colors['edge'], width='0.4', fixedsize='true')
        
        # Pre-FFN Norm
        b.node('LN2', 'RMSNorm', shape='box', style='rounded,filled', fillcolor='white', color=colors['edge'], width='1.5')

        # MoE Block (Detailed Card)
        # We split MoE visual into logic parts for clarity, but keep style consistent
        
        # Router
        b.node('Router', 'Router (Gate)', shape='component', style='filled', fillcolor=colors['router'], color='#ffcc80')
        
        # Shared Path
        shared_details = {"Count": "1", "Type": "Always Active"}
        b.node('Shared', make_card("Shared Experts", "Base Knowledge", shared_details, '#c8e6c9', '#a5d6a7', width=120))
        
        # Routed Path
        routed_details = {"Total": "4", "Active (Top-K)": "2", "Hidden Dim": "512"}
        b.node('Routed', make_card("Routed Experts", "Specialized FFN", routed_details, colors['moe_header'], colors['moe_border'], width=150))
        
        # Aggregation
        b.node('Sum', 'Weighted\nSum', shape='diamond', style='filled', fillcolor='white', fontsize='9', height='0.6')

        # Residual 2
        b.node('Add2', '+', shape='circle', style='filled', fillcolor='white', color=colors['edge'], width='0.4', fixedsize='true')

        # -- Edges inside block --
        b.edge('LN1', 'Attn')
        b.edge('Attn', 'Add1')
        b.edge('LN2', 'Router')
        
        # Split paths
        b.edge('LN2', 'Shared', style='solid')
        b.edge('LN2', 'Routed', style='solid')
        
        # Routing logic
        b.edge('Router', 'Routed', label=' <font point-size="9" color="#e65100">Select Top-2</font>', color='#ffb74d')
        
        # Rejoin
        b.edge('Shared', 'Sum')
        b.edge('Routed', 'Sum')
        b.edge('Sum', 'Add2')

    # --- 3. Connections & Residuals ---
    dot.edge('In', 'LN1')
    dot.edge('In', 'Add1', xlabel='residual ', style='dashed', color='#b0bec5')
    dot.edge('Add1', 'LN2')
    dot.edge('Add1', 'Add2', xlabel='residual ', style='dashed', color='#b0bec5')
    
    # --- 4. Output ---
    dot.node('Out', make_card(
        "Output Logits", 
        "Next Token Prediction", 
        {"Vocab Size": "10,000", "Shape": "(B, T, V)"}, 
        colors['input'], colors['block_border']
    ))
    dot.edge('Add2', 'Out')

    dot.render('pretty_architecture', view=False)
    print("Generated aesthetic diagram: pretty_architecture.png")

if __name__ == "__main__":
    draw_aesthetic_architecture()