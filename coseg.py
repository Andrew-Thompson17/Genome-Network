import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

file_path = 'C:\\compsci\\4150\\coseg\\GSE64881_segmentation_at_30000bp.passqc.multibam.txt'

# Read the text file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')
# Hist1 Region
filtered_data = df[(df['chrom'] == 'chr13') & (df['start'] > 21680000) & (df['stop'] < 24140000)]
ignored_columns = filtered_data.iloc[:, 3:].loc[:, (filtered_data.iloc[:, 3:] != 0).any()]
# Transpose 
df_trans = ignored_columns.transpose()
print(df_trans.head())
# Step 2
#Calculate normalized linkage
def calc_norm_link(seg1, seg2):
    fab = 0
    #fab = (np.sum(seg1) + np.sum(seg2)) / len(seg1)
    for i in range (len(seg1)):
        if seg1[i] == 1 and seg2[i] == 1:
            fab += 1
    fab = fab / len(seg1)
    fa = np.mean(seg1)
    fb = np.mean(seg2)
    link = fab - (fa * fb)

    # Calculate Dmax
    if link < 0:
        Dmax = min(fab, (1 - fa) * (1 - fb))
    elif link > 0:
        Dmax = min(fb * (1 - fa), fa * (1 - fb))
    else:
        return 0
    
    # Handle division by zero
    if Dmax != 0:
        return link / Dmax
    else:
        return 0

# Normalized linkage table
num_segments = len(df_trans.columns)
normalized_linkage_table = np.zeros((num_segments, num_segments))

# Iterate over segments to calculate normalized linkage
for i, (name1, segment1) in enumerate(df_trans.items()):
    for j, (name2, segment2) in enumerate(df_trans.items()):
        normalized_linkage_table[i, j] = calc_norm_link(segment1, segment2)

print(normalized_linkage_table)

# Create a heatmap figure
fig = go.Figure(data=go.Heatmap(z=normalized_linkage_table, colorscale='sunset'))

# Customize layout
fig.update_layout(
    title='Normalized Linkage Heatmap',
    xaxis=dict(title='Segments'),
    yaxis=dict(title='Segments'),
    coloraxis_colorbar=dict(title='Normalized Linkage')
)

# Save the figure as an HTML file
fig.write_html('normalized_linkage_heatmap.html')

#Part 2
# If linkage is less than average linkage, it is not an edge.
# .draw() should make the figure

#Use np.array min mean max

def create_edge_graph(normalized_linkage_table):
    l_avg = np.mean(normalized_linkage_table)
    num_segments = len(normalized_linkage_table)
    edge_matrix = [[0] * num_segments for _ in range(num_segments)]  # Initialize with zeros
    edge_graph = nx.Graph()
    edge_graph.add_nodes_from([i for i in range(num_segments)])
    for i in range(num_segments):
        for j in range(i+1, num_segments):
            if l_avg < normalized_linkage_table[i, j]:
                edge_graph.add_edge(i, j)
                edge_matrix[i][j] = 1
    return edge_graph, edge_matrix


def calc_centrality(edge_matrix):
    row_sum = edge_matrix.sum(axis = 1)
    return row_sum / (len(edge_matrix)-1)

# Create the edge graph and edge matrix
edge_graph, edge_matrix = create_edge_graph(normalized_linkage_table)

# Calculate centrality
centrality = calc_centrality(np.array(edge_matrix))

# Get min, max, and average centrality
min_centrality = np.min(centrality)
max_centrality = np.max(centrality)
avg_centrality = np.mean(centrality)

print("Minimum Centrality:", min_centrality)
print("Maximum Centrality:", max_centrality)
print("Average Centrality:", avg_centrality)

edge_graph, _ = create_edge_graph(normalized_linkage_table)

def find_hubs_with_communities(edge_graph):
    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(edge_graph)

    # Sort nodes by degree centrality in descending order
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # Select top 5 nodes with highest degree centrality
    top_hubs = sorted_nodes[:5]

    # Define communities for each top hub
    hub_communities = []
    for hub, _ in top_hubs:
        community = set([hub])  # Initialize community with hub
        neighbors = set(edge_graph.neighbors(hub))  # Get neighbors of hub
        community |= neighbors  # Include neighbors in the community
        hub_communities.append(community)

    return top_hubs, hub_communities

# check if a given node has a value greater than 0 in 'Hist1' and 'LAD' columns
def check_hist1_and_lad(node, feature_table):
    hist1_value = feature_table.at[node, 'Hist1']
    lad_value = feature_table.at[node, 'LAD']
    return hist1_value > 0, lad_value > 0

def analyze_hubs_with_communities(edge_graph, filename):
    # Find hubs with communities
    top_hubs, hub_communities = find_hubs_with_communities(edge_graph)
    
    # Analyze communities
    for i, (hub, community) in enumerate(zip(top_hubs, hub_communities), 1):
        hub_node, _ = hub
        print(f"Community {i} (Hub #{hub_node}):")  # Print hub number
        community_size = len(community)
        nodes_list = list(community)
        print(f"   Size of the community: {community_size}")
        print("   List of nodes in the community:", nodes_list)
        print()
        
        # Read feature table
        feature_table = pd.read_csv(filename, delimiter=",")
        
        # Check Hist1 and LAD for each node in the community
        hist1_counts = 0
        lad_counts = 0
        for node in community:
            hist1_exists, lad_exists = check_hist1_and_lad(node, feature_table)
            if hist1_exists:
                hist1_counts += 1
            if lad_exists:
                lad_counts += 1
        
        # Calculate percentages
        hist1_percentage = hist1_counts / len(community) * 100
        lad_percentage = lad_counts / len(community) * 100
        
        print(f"   Percentage of nodes in the community that contain a Hist1 gene: {hist1_percentage:.2f}%")
        print(f"   Percentage of nodes in the community that contain a LAD: {lad_percentage:.2f}%")
        print()


# Call the combined function
filename = "Hist1_region_features.csv"
analyze_hubs_with_communities(edge_graph, filename)

top_hubs, hub_communities = find_hubs_with_communities(edge_graph)

def visualize_community_heatmap_and_graph(edge_graph, communities, centrality):
    for i, community in enumerate(communities, 1):
        # Create a subgraph for the community
        community_subgraph = edge_graph.subgraph(community)

        # Get adjacency matrix for the subgraph
        adjacency_matrix = nx.adjacency_matrix(community_subgraph).todense()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(adjacency_matrix, cmap='viridis', square=True, cbar=False)
        plt.title(f'Community {i} Heatmap')
        plt.xlabel('Genomic Window Index')
        plt.ylabel('Genomic Window Index')
        plt.savefig(f'community_{i}_heatmap.png')  # Save heatmap as PNG
        plt.close()  # Close the current figure to prevent overlapping
        
        # Plot the graph with varying node sizes based on centrality
        plt.figure(figsize=(10, 8))
        node_sizes = [centrality[node] * 1000 for node in community]  # Scale centrality to adjust node size
        nx.draw(community_subgraph, with_labels=True, node_color='skyblue', node_size=node_sizes, edge_color='black', linewidths=1, font_size=10)
        plt.title(f'Community {i} Graph')
        plt.savefig(f'community_{i}_graph.png')  # Save graph as PNG
        plt.close()  # Close the current figure to prevent overlapping

# Call the function to visualize the heatmap and graph for each community
visualize_community_heatmap_and_graph(edge_graph, hub_communities, centrality)

# Writing the output to a Markdown file
with open('output.md', 'w') as f:
    f.write("# Output\n\n")
    
    f.write("## Normalized Linkage Heatmap\n\n")
    f.write("![Normalized Linkage Heatmap](normalized_linkage_heatmap.html)\n\n")
    
    for i, community in enumerate(hub_communities, 1):
        f.write(f"## Community {i}\n\n")
        f.write(f"### Heatmap\n\n")
        f.write(f"![Community {i} Heatmap](community_{i}_heatmap.png)\n\n")
        f.write(f"### Graph\n\n")
        f.write(f"![Community {i} Graph](community_{i}_graph.png)\n\n")

print("Output written to output.md")

##plt.figure(figsize=(10, 8))
##nx.draw(edge_graph, with_labels=True, node_color='skyblue', node_size=500, edge_color='black', linewidths=1, font_size=10)
##plt.title('Edge Graph')
##plt.show()
