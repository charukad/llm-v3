import plotly.graph_objects as go
import numpy as np
import os
import json
import uuid
from typing import Dict, List, Tuple, Optional, Union, Any

class LinearAlgebraVisualizer:
    """
    Class for generating visualizations related to linear algebra concepts.
    """
    
    def __init__(self, 
                 output_dir: str = "static/visualizations",
                 img_format: str = "png",
                 save_html: bool = True,
                 default_width: int = 800,
                 default_height: int = 800):
        """
        Initialize the linear algebra visualizer.
        
        Args:
            output_dir: Directory to save generated plots
            img_format: Format for static image exports ("png" or "svg")
            save_html: Whether to save interactive HTML versions
            default_width: Default width for plots
            default_height: Default height for plots
        """
        self.output_dir = output_dir
        self.img_format = img_format
        self.save_html = save_html
        self.default_width = default_width
        self.default_height = default_height
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        if save_html:
            os.makedirs(os.path.join(output_dir, "html"), exist_ok=True)
    
    def visualize_matrix_transformation(self,
                                      matrix: Union[List[List[float]], np.ndarray],
                                      title: Optional[str] = None,
                                      show_eigenvectors: bool = True,
                                      grid_size: int = 5,
                                      grid_spacing: float = 1.0,
                                      filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize a linear transformation represented by a 2x2 or 3x3 matrix.
        
        Args:
            matrix: 2x2 or 3x3 transformation matrix
            title: Plot title
            show_eigenvectors: Whether to show eigenvectors
            grid_size: Size of the grid to transform
            grid_spacing: Spacing between grid lines
            filename: Filename to save the plot (without extension)
            
        Returns:
            Dictionary with plot result
        """
        try:
            # Convert to numpy array
            matrix = np.array(matrix, dtype=float)
            
            # Check matrix dimensions
            if matrix.shape not in [(2, 2), (3, 3)]:
                return {
                    'success': False,
                    'file_path': None,
                    'html_path': None,
                    'error': f'Matrix must be 2x2 or 3x3, got {matrix.shape}',
                    'metadata': None
                }
            
            # Determine if we're working in 2D or 3D
            is_3d = matrix.shape == (3, 3)
            
            # Create a grid of points to visualize the transformation
            if is_3d:
                # Create 3D grid
                x = np.arange(-grid_size, grid_size + 1, grid_spacing)
                y = np.arange(-grid_size, grid_size + 1, grid_spacing)
                z = np.arange(-grid_size, grid_size + 1, grid_spacing)
                
                # Create figure
                fig = go.Figure()
                
                # Add transformed x, y, and z axes lines
                self._add_transformed_axes_3d(fig, matrix, grid_size)
                
                # Add transformed grid planes
                self._add_transformed_grid_planes(fig, matrix, grid_size, grid_spacing)
                
                # Add unit cube transformation
                self._add_transformed_unit_cube(fig, matrix)
                
                # Add eigenvectors if requested
                if show_eigenvectors:
                    self._add_eigenvectors_3d(fig, matrix, grid_size)
                
                # Update layout
                fig.update_layout(
                    title=title if title else "3D Matrix Transformation",
                    autosize=False,
                    width=self.default_width,
                    height=self.default_height,
                    scene=dict(
                        xaxis=dict(range=[-grid_size, grid_size], title='x'),
                        yaxis=dict(range=[-grid_size, grid_size], title='y'),
                        zaxis=dict(range=[-grid_size, grid_size], title='z'),
                        aspectratio=dict(x=1, y=1, z=1)
                    ),
                    margin=dict(l=50, r=50, b=50, t=80)
                )
                
            else:
                # Create 2D grid
                x = np.arange(-grid_size, grid_size + 1, grid_spacing)
                y = np.arange(-grid_size, grid_size + 1, grid_spacing)
                X, Y = np.meshgrid(x, y)
                
                # Create points for original grid
                original_points = np.vstack([X.flatten(), Y.flatten()])
                
                # Apply transformation
                transformed_points = matrix @ original_points
                
                # Reshape for plotting
                X_transformed = transformed_points[0, :].reshape(X.shape)
                Y_transformed = transformed_points[1, :].reshape(Y.shape)
                
                # Create figure
                fig = go.Figure()
                
                # Add original grid lines
                for i in range(len(x)):
                    fig.add_trace(go.Scatter(
                        x=X[i, :], y=Y[i, :],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        name=f'Original Horizontal {i}' if i == 0 else '',
                        showlegend=i == 0
                    ))
                
                for j in range(len(y)):
                    fig.add_trace(go.Scatter(
                        x=X[:, j], y=Y[:, j],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        name=f'Original Vertical {j}' if j == 0 else '',
                        showlegend=j == 0
                    ))
                
                # Add transformed grid lines
                for i in range(len(x)):
                    fig.add_trace(go.Scatter(
                        x=X_transformed[i, :], y=Y_transformed[i, :],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        name=f'Transformed Horizontal {i}' if i == 0 else '',
                        showlegend=i == 0
                    ))
                
                for j in range(len(y)):
                    fig.add_trace(go.Scatter(
                        x=X_transformed[:, j], y=Y_transformed[:, j],
                        mode='lines',
                        line=dict(color='blue', width=1),
                        name=f'Transformed Vertical {j}' if j == 0 else '',
                        showlegend=j == 0
                    ))
                
                # Add original axes
                fig.add_trace(go.Scatter(
                    x=[-grid_size, grid_size], y=[0, 0],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Original x-axis'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[-grid_size, grid_size],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Original y-axis'
                ))
                
                # Add transformed axes
                fig.add_trace(go.Scatter(
                    x=[0, matrix[0, 0] * grid_size], y=[0, matrix[1, 0] * grid_size],
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Transformed x-axis'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, matrix[0, 1] * grid_size], y=[0, matrix[1, 1] * grid_size],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name='Transformed y-axis'
                ))
                
                # Add unit vectors
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 0],
                    mode='lines+markers',
                    marker=dict(size=8, color='red'),
                    line=dict(color='red', width=3),
                    name='Unit vector i'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[0, 1],
                    mode='lines+markers',
                    marker=dict(size=8, color='green'),
                    line=dict(color='green', width=3),
                    name='Unit vector j'
                ))
                
                # Add transformed unit vectors
                fig.add_trace(go.Scatter(
                    x=[0, matrix[0, 0]], y=[0, matrix[1, 0]],
                    mode='lines+markers',
                    marker=dict(size=8, color='red'),
                    line=dict(color='red', width=3, dash='dash'),
                    name='Transformed i'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, matrix[0, 1]], y=[0, matrix[1, 1]],
                    mode='lines+markers',
                    marker=dict(size=8, color='green'),
                    line=dict(color='green', width=3, dash='dash'),
                    name='Transformed j'
                ))
                
                # Add eigenvectors if requested
                if show_eigenvectors:
                    eigenvalues, eigenvectors = np.linalg.eig(matrix)
                    
                    for i in range(len(eigenvalues)):
                        if np.isclose(eigenvalues[i].imag, 0):  # Only show real eigenvectors
                            ev = eigenvectors[:, i].real
                            lambda_val = eigenvalues[i].real
                            
                            # Normalize eigenvector for visualization
                            ev_norm = ev / np.linalg.norm(ev) * 2
                            
                            fig.add_trace(go.Scatter(
                                x=[0, ev_norm[0]], y=[0, ev_norm[1]],
                                mode='lines',
                                line=dict(color='purple', width=3),
                                name=f'Eigenvector {i+1} (λ={lambda_val:.2f})'
                            ))
                            
                            # Show transformed eigenvector (should be parallel)
                            transformed_ev = matrix @ ev_norm
                            fig.add_trace(go.Scatter(
                                x=[0, transformed_ev[0]], y=[0, transformed_ev[1]],
                                mode='lines',
                                line=dict(color='purple', width=3, dash='dash'),
                                name=f'Transformed EV {i+1}'
                            ))
                
                # Update layout
                fig.update_layout(
                    title=title if title else "2D Matrix Transformation",
                    autosize=False,
                    width=self.default_width,
                    height=self.default_height,
                    xaxis=dict(
                        range=[-grid_size, grid_size],
                        constrain="domain",
                        scaleanchor="y"
                    ),
                    yaxis=dict(
                        range=[-grid_size, grid_size],
                        constrain="domain"
                    ),
                    margin=dict(l=50, r=50, b=50, t=80)
                )
                
                # Add matrix annotation
                matrix_text = [
                    f"[{matrix[0, 0]:.2f}, {matrix[0, 1]:.2f}]",
                    f"[{matrix[1, 0]:.2f}, {matrix[1, 1]:.2f}]"
                ]
                
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text="Matrix:<br>" + "<br>".join(matrix_text),
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )
                
                # Calculate and add determinant
                determinant = np.linalg.det(matrix)
                fig.add_annotation(
                    x=0.02,
                    y=0.82,
                    xref="paper",
                    yref="paper",
                    text=f"Det = {determinant:.2f}",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )
            
            # Generate filename if not provided
            if not filename:
                dim_str = "3d" if is_3d else "2d"
                filename = f"matrix_transform_{dim_str}_{uuid.uuid4().hex[:8]}"
            
            # Save as image
            img_path = os.path.join(self.output_dir, f"{filename}.{self.img_format}")
            fig.write_image(img_path)
            
            # Save as HTML if requested
            html_path = None
            if self.save_html:
                html_path = os.path.join(self.output_dir, "html", f"{filename}.html")
                fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Return success result
            return {
                'success': True,
                'file_path': img_path,
                'html_path': html_path,
                'error': None,
                'metadata': {
                    'matrix': matrix.tolist(),
                    'determinant': float(np.linalg.det(matrix)),
                    'eigenvalues': [float(v.real) if np.isclose(v.imag, 0) else complex(v) for v in np.linalg.eigvals(matrix)],
                    'title': title,
                    'is_3d': is_3d,
                    'plot_type': '3d_transform' if is_3d else '2d_transform'
                }
            }
            
        except Exception as e:
            # Return failure result with error message
            return {
                'success': False,
                'file_path': None,
                'html_path': None,
                'error': str(e),
                'metadata': None
            }
    
    def visualize_eigenvectors(self,
                              matrix: Union[List[List[float]], np.ndarray],
                              title: Optional[str] = None,
                              grid_size: int = 5,
                              show_eigenspaces: bool = True,
                              filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize the eigenvectors of a 2x2 or 3x3 matrix.
        
        Args:
            matrix: 2x2 or 3x3 matrix
            title: Plot title
            grid_size: Size of the grid for visualization
            show_eigenspaces: Whether to show eigenspaces (lines or planes)
            filename: Filename to save the plot (without extension)
            
        Returns:
            Dictionary with plot result
        """
        try:
            # Convert to numpy array
            matrix = np.array(matrix, dtype=float)
            
            # Check matrix dimensions
            if matrix.shape not in [(2, 2), (3, 3)]:
                return {
                    'success': False,
                    'file_path': None,
                    'html_path': None,
                    'error': f'Matrix must be 2x2 or 3x3, got {matrix.shape}',
                    'metadata': None
                }
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            # Determine if we're working in 2D or 3D
            is_3d = matrix.shape == (3, 3)
            
            # Create figure
            fig = go.Figure()
            
            if is_3d:
                # Add coordinate axes
                self._add_axes_3d(fig, grid_size)
                
                # Add eigenvectors
                for i in range(len(eigenvalues)):
                    ev = eigenvectors[:, i]
                    lambda_val = eigenvalues[i]
                    
                    # Skip complex eigenvectors for now
                    if np.isclose(lambda_val.imag, 0) and np.allclose(ev.imag, 0):
                        # Normalize eigenvector for better visualization
                        ev_real = ev.real
                        ev_norm = ev_real / np.linalg.norm(ev_real) * grid_size
                        
                        # Add both directions of eigenvector
                        fig.add_trace(go.Scatter3d(
                            x=[0, ev_norm[0]],
                            y=[0, ev_norm[1]],
                            z=[0, ev_norm[2]],
                            mode='lines',
                            line=dict(color=self._get_color(i), width=6),
                            name=f'Eigenvector {i+1} (λ={lambda_val.real:.2f})'
                        ))
                        
                        fig.add_trace(go.Scatter3d(
                            x=[0, -ev_norm[0]],
                            y=[0, -ev_norm[1]],
                            z=[0, -ev_norm[2]],
                            mode='lines',
                            line=dict(color=self._get_color(i), width=6, dash='dash'),
                            showlegend=False
                        ))
                        
                        # Add eigenspace (plane) if requested
                        if show_eigenspaces:
                            # Generate a meshgrid for the eigenspace (plane orthogonal to eigenvector)
                            # First, find two orthogonal vectors to the eigenvector
                            ortho1, ortho2 = self._find_orthogonal_vectors(ev_real)
                            
                            # Scale and generate grid
                            scaled_ortho1 = ortho1 * grid_size / 2
                            scaled_ortho2 = ortho2 * grid_size / 2
                            
                            # Create a grid of points for the plane
                            u = np.linspace(-1, 1, 10)
                            v = np.linspace(-1, 1, 10)
                            U, V = np.meshgrid(u, v)
                            
                            X = U * scaled_ortho1[0] + V * scaled_ortho2[0]
                            Y = U * scaled_ortho1[1] + V * scaled_ortho2[1]
                            Z = U * scaled_ortho1[2] + V * scaled_ortho2[2]
                            
                            fig.add_trace(go.Surface(
                                x=X, y=Y, z=Z,
                                colorscale=[[0, self._get_color(i)], [1, self._get_color(i)]],
                                opacity=0.3,
                                showscale=False,
                                name=f'Eigenspace {i+1}'
                            ))
                
                # Update layout
                fig.update_layout(
                    title=title if title else "3D Eigenvectors Visualization",
                    autosize=False,
                    width=self.default_width,
                    height=self.default_height,
                    scene=dict(
                        xaxis=dict(range=[-grid_size, grid_size], title='x'),
                        yaxis=dict(range=[-grid_size, grid_size], title='y'),
                        zaxis=dict(range=[-grid_size, grid_size], title='z'),
                        aspectratio=dict(x=1, y=1, z=1)
                    ),
                    margin=dict(l=50, r=50, b=50, t=80)
                )
                
            else:  # 2D case
                # Add coordinate axes
                fig.add_trace(go.Scatter(
                    x=[-grid_size, grid_size], y=[0, 0],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name='x-axis'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 0], y=[-grid_size, grid_size],
                    mode='lines',
                    line=dict(color='black', width=1),
                    name='y-axis'
                ))
                
                # Add eigenvectors
                for i in range(len(eigenvalues)):
                    ev = eigenvectors[:, i]
                    lambda_val = eigenvalues[i]
                    
                    # Skip complex eigenvectors for now
                    if np.isclose(lambda_val.imag, 0) and np.allclose(ev.imag, 0):
                        # Normalize eigenvector for better visualization
                        ev_real = ev.real
                        ev_norm = ev_real / np.linalg.norm(ev_real) * grid_size
                        
                        # Add both directions of eigenvector
                        fig.add_trace(go.Scatter(
                            x=[0, ev_norm[0]],
                            y=[0, ev_norm[1]],
                            mode='lines+markers',
                            line=dict(color=self._get_color(i), width=3),
                            marker=dict(size=8, color=self._get_color(i)),
                            name=f'Eigenvector {i+1} (λ={lambda_val.real:.2f})'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[0, -ev_norm[0]],
                            y=[0, -ev_norm[1]],
                            mode='lines',
                            line=dict(color=self._get_color(i), width=3, dash='dash'),
                            showlegend=False
                        ))
                        
                        # Add transformed eigenvector
                        transformed_ev = matrix @ ev_norm
                        fig.add_trace(go.Scatter(
                            x=[0, transformed_ev[0]],
                            y=[0, transformed_ev[1]],
                            mode='lines',
                            line=dict(color=self._get_color(i), width=3, dash='dot'),
                            name=f'Transformed EV {i+1} (= λ⋅v)'
                        ))
                        
                        # Add eigenspace (line) if requested and eigenvalue is not zero
                        if show_eigenspaces and not np.isclose(lambda_val.real, 0):
                            # For a 2D eigenspace, just extend the eigenvector in both directions
                            fig.add_trace(go.Scatter(
                                x=[-ev_norm[0] * 2, ev_norm[0] * 2],
                                y=[-ev_norm[1] * 2, ev_norm[1] * 2],
                                mode='lines',
                                line=dict(color=self._get_color(i), width=1),
                                name=f'Eigenspace {i+1}'
                            ))
                
                # Update layout
                fig.update_layout(
                    title=title if title else "2D Eigenvectors Visualization",
                    autosize=False,
                    width=self.default_width,
                    height=self.default_height,
                    xaxis=dict(
                        range=[-grid_size, grid_size],
                        constrain="domain",
                        scaleanchor="y"
                    ),
                    yaxis=dict(
                        range=[-grid_size, grid_size],
                        constrain="domain"
                    ),
                    margin=dict(l=50, r=50, b=50, t=80)
                )
                
                # Add matrix annotation
                matrix_text = [
                    f"[{matrix[0, 0]:.2f}, {matrix[0, 1]:.2f}]",
                    f"[{matrix[1, 0]:.2f}, {matrix[1, 1]:.2f}]"
                ]
                
                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text="Matrix:<br>" + "<br>".join(matrix_text),
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=2
                )
                
                # Add eigenvalues annotation
                eigenvalues_text = [f"λ{i+1} = {ev:.2f}" for i, ev in enumerate(eigenvalues) if np.isclose(ev.imag, 0)]
                
                if eigenvalues_text:
                    fig.add_annotation(
                        x=0.02,
                        y=0.82,
                        xref="paper",
                        yref="paper",
                        text="Eigenvalues:<br>" + "<br>".join(eigenvalues_text),
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="rgba(255, 255, 255, 0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=2
                    )
            
            # Generate filename if not provided
            if not filename:
                dim_str = "3d" if is_3d else "2d"
                filename = f"eigenvectors_{dim_str}_{uuid.uuid4().hex[:8]}"
            
            # Save as image
            img_path = os.path.join(self.output_dir, f"{filename}.{self.img_format}")
            fig.write_image(img_path)
            
            # Save as HTML if requested
            html_path = None
            if self.save_html:
                html_path = os.path.join(self.output_dir, "html", f"{filename}.html")
                fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Calculate eigenvalue statistics
            eig_stats = {
                'real_count': sum(1 for ev in eigenvalues if np.isclose(ev.imag, 0)),
                'complex_count': sum(1 for ev in eigenvalues if not np.isclose(ev.imag, 0)),
                'positive_count': sum(1 for ev in eigenvalues if np.isclose(ev.imag, 0) and ev.real > 0),
                'negative_count': sum(1 for ev in eigenvalues if np.isclose(ev.imag, 0) and ev.real < 0),
                'zero_count': sum(1 for ev in eigenvalues if np.isclose(ev.imag, 0) and np.isclose(ev.real, 0))
            }
            
            # Return success result
            return {
                'success': True,
                'file_path': img_path,
                'html_path': html_path,
                'error': None,
                'metadata': {
                    'matrix': matrix.tolist(),
                    'eigenvalues': [float(v.real) if np.isclose(v.imag, 0) else complex(v) for v in eigenvalues],
                    'eigenvectors': [v.real.tolist() if np.allclose(v.imag, 0) else v.tolist() for v in eigenvectors.T],
                    'eigenvalue_stats': eig_stats,
                    'title': title,
                    'is_3d': is_3d,
                    'plot_type': '3d_eigenvectors' if is_3d else '2d_eigenvectors'
                }
            }
            
        except Exception as e:
            # Return failure result with error message
            return {
                'success': False,
                'file_path': None,
                'html_path': None,
                'error': str(e),
                'metadata': None
            }
    
    def visualize_matrix_heatmap(self,
                                matrix: Union[List[List[float]], np.ndarray],
                                title: Optional[str] = None,
                                colorscale: str = "Viridis",
                                show_values: bool = True,
                                row_labels: Optional[List[str]] = None,
                                col_labels: Optional[List[str]] = None,
                                filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize a matrix as a heatmap.
        
        Args:
            matrix: Matrix to visualize
            title: Plot title
            colorscale: Colorscale for the heatmap
            show_values: Whether to show cell values
            row_labels: Labels for rows
            col_labels: Labels for columns
            filename: Filename to save the plot (without extension)
            
        Returns:
            Dictionary with plot result
        """
        try:
            # Convert to numpy array
            matrix = np.array(matrix, dtype=float)
            
            # Check matrix dimensions
            if len(matrix.shape) != 2:
                return {
                    'success': False,
                    'file_path': None,
                    'html_path': None,
                    'error': f'Input must be a 2D matrix, got shape {matrix.shape}',
                    'metadata': None
                }
            
            # Get matrix dimensions
            n_rows, n_cols = matrix.shape
            
            # Create default labels if not provided
            if row_labels is None:
                row_labels = [f"Row {i+1}" for i in range(n_rows)]
            
            if col_labels is None:
                col_labels = [f"Col {i+1}" for i in range(n_cols)]
            
            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=col_labels,
                y=row_labels,
                colorscale=colorscale,
                hoverongaps=False
            ))
            
            # Add values as text if requested
            if show_values:
                text_matrix = [[f"{val:.2f}" for val in row] for row in matrix]
                
                fig.add_trace(go.Heatmap(
                    z=matrix,
                    x=col_labels,
                    y=row_labels,
                    text=text_matrix,
                    texttemplate="%{text}",
                    showscale=False,
                    hoverinfo="none",
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                ))
            
            # Update layout
            fig.update_layout(
                title=title if title else "Matrix Heatmap",
                autosize=False,
                width=self.default_width,
                height=self.default_height,
                margin=dict(l=50, r=50, b=50, t=80)
            )
            
            # Calculate matrix properties
            matrix_properties = {
                'determinant': float(np.linalg.det(matrix)) if n_rows == n_cols else None,
                'trace': float(np.trace(matrix)) if n_rows == n_cols else None,
                'rank': int(np.linalg.matrix_rank(matrix)),
                'condition_number': float(np.linalg.cond(matrix)) if n_rows == n_cols else None,
                'is_symmetric': bool(np.allclose(matrix, matrix.T)) if n_rows == n_cols else False,
                'is_invertible': bool(np.linalg.det(matrix) != 0) if n_rows == n_cols else False,
                'eigenvalues': [float(v.real) if np.isclose(v.imag, 0) else complex(v) for v in np.linalg.eigvals(matrix)] if n_rows == n_cols else None
            }
            
            # Generate filename if not provided
            if not filename:
                filename = f"matrix_heatmap_{uuid.uuid4().hex[:8]}"
            
            # Save as image
            img_path = os.path.join(self.output_dir, f"{filename}.{self.img_format}")
            fig.write_image(img_path)
            
            # Save as HTML if requested
            html_path = None
            if self.save_html:
                html_path = os.path.join(self.output_dir, "html", f"{filename}.html")
                fig.write_html(html_path, include_plotlyjs='cdn')
            
            # Return success result
            return {
                'success': True,
                'file_path': img_path,
                'html_path': html_path,
                'error': None,
                'metadata': {
                    'matrix': matrix.tolist(),
                    'properties': matrix_properties,
                    'title': title,
                    'dimensions': matrix.shape,
                    'plot_type': 'matrix_heatmap'
                }
            }
            
        except Exception as e:
            # Return failure result with error message
            return {
                'success': False,
                'file_path': None,
                'html_path': None,
                'error': str(e),
                'metadata': None
            }
    
    # Helper methods
    def _get_color(self, index: int) -> str:
        """Get a color from a predefined color cycle."""
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        return colors[index % len(colors)]
    
    def _add_axes_3d(self, fig, grid_size: float):
        """Add coordinate axes to a 3D figure."""
        # Add x-axis
        fig.add_trace(go.Scatter3d(
            x=[-grid_size, grid_size],
            y=[0, 0],
            z=[0, 0],
            mode='lines',
            line=dict(color='red', width=3),
            name='x-axis'
        ))
        
        # Add y-axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[-grid_size, grid_size],
            z=[0, 0],
            mode='lines',
            line=dict(color='green', width=3),
            name='y-axis'
        ))
        
        # Add z-axis
        fig.add_trace(go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-grid_size, grid_size],
            mode='lines',
            line=dict(color='blue', width=3),
            name='z-axis'
        ))
    
    def _add_transformed_axes_3d(self, fig, matrix, grid_size: float):
        """Add transformed coordinate axes to a 3D figure."""
        # Original axes
        self._add_axes_3d(fig, grid_size)
        
        # X-axis direction vector and its transformation
        x_vec = np.array([1, 0, 0]) * grid_size
        transformed_x = matrix @ x_vec
        
        # Y-axis direction vector and its transformation
        y_vec = np.array([0, 1, 0]) * grid_size
        transformed_y = matrix @ y_vec
        
        # Z-axis direction vector and its transformation
        z_vec = np.array([0, 0, 1]) * grid_size
        transformed_z = matrix @ z_vec
        
        # Add transformed x-axis
        fig.add_trace(go.Scatter3d(
            x=[0, transformed_x[0]],
            y=[0, transformed_x[1]],
            z=[0, transformed_x[2]],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            name='Transformed x-axis'
        ))
        
        # Add transformed y-axis
        fig.add_trace(go.Scatter3d(
            x=[0, transformed_y[0]],
            y=[0, transformed_y[1]],
            z=[0, transformed_y[2]],
            mode='lines',
            line=dict(color='green', width=3, dash='dash'),
            name='Transformed y-axis'
        ))
        
        # Add transformed z-axis
        fig.add_trace(go.Scatter3d(
            x=[0, transformed_z[0]],
            y=[0, transformed_z[1]],
            z=[0, transformed_z[2]],
            mode='lines',
            line=dict(color='blue', width=3, dash='dash'),
            name='Transformed z-axis'
        ))
    
    def _add_transformed_grid_planes(self, fig, matrix, grid_size: float, grid_spacing: float):
        """Add transformed grid planes to a 3D figure."""
        # Create grid in xy-plane
        x = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
        y = np.arange(-grid_size, grid_size + grid_spacing, grid_spacing)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Create points for the grid
        points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # Apply transformation
        transformed_points = matrix @ points
        
        # Reshape for plotting
        X_transformed = transformed_points[0, :].reshape(X.shape)
        Y_transformed = transformed_points[1, :].reshape(Y.shape)
        Z_transformed = transformed_points[2, :].reshape(Z.shape)
        
        # Add transformed xy-plane grid
        for i in range(len(x)):
            fig.add_trace(go.Scatter3d(
                x=X_transformed[i, :],
                y=Y_transformed[i, :],
                z=Z_transformed[i, :],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=i == 0,
                name='Transformed xy-grid' if i == 0 else None
            ))
        
        for j in range(len(y)):
            fig.add_trace(go.Scatter3d(
                x=X_transformed[:, j],
                y=Y_transformed[:, j],
                z=Z_transformed[:, j],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
    
    def _add_transformed_unit_cube(self, fig, matrix):
        """Add a transformed unit cube to a 3D figure."""
        # Unit cube vertices
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ]).T
        
        # Apply transformation
        transformed_vertices = matrix @ vertices
        
        # Edges of the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[transformed_vertices[0, edge[0]], transformed_vertices[0, edge[1]]],
                y=[transformed_vertices[1, edge[0]], transformed_vertices[1, edge[1]]],
                z=[transformed_vertices[2, edge[0]], transformed_vertices[2, edge[1]]],
                mode='lines',
                line=dict(color='black', width=3),
                showlegend=edge == edges[0],
                name='Transformed Unit Cube' if edge == edges[0] else None
            ))
    
    def _add_eigenvectors_3d(self, fig, matrix, grid_size: float):
        """Add eigenvectors to a 3D figure."""
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # Add eigenvectors
        for i in range(len(eigenvalues)):
            ev = eigenvectors[:, i]
            lambda_val = eigenvalues[i]
            
            # Skip complex eigenvectors
            if np.isclose(lambda_val.imag, 0) and np.allclose(ev.imag, 0):
                # Normalize eigenvector for better visualization
                ev_real = ev.real
                ev_norm = ev_real / np.linalg.norm(ev_real) * grid_size / 2
                
                # Add eigenvector
                fig.add_trace(go.Scatter3d(
                    x=[0, ev_norm[0]],
                    y=[0, ev_norm[1]],
                    z=[0, ev_norm[2]],
                    mode='lines',
                    line=dict(color=self._get_color(i), width=5),
                    name=f'Eigenvector {i+1} (λ={lambda_val.real:.2f})'
                ))
                
                # Add transformed eigenvector (should be parallel, scaled by eigenvalue)
                transformed_ev = lambda_val.real * ev_norm
                fig.add_trace(go.Scatter3d(
                    x=[0, transformed_ev[0]],
                    y=[0, transformed_ev[1]],
                    z=[0, transformed_ev[2]],
                    mode='lines',
                    line=dict(color=self._get_color(i), width=5, dash='dash'),
                    name=f'Transformed EV {i+1}'
                ))
    
    def _find_orthogonal_vectors(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find two orthogonal vectors to the given vector v.
        
        Args:
            v: Vector for which to find orthogonal vectors
            
        Returns:
            Tuple of two orthogonal unit vectors
        """
        # Normalize input vector
        v = v / np.linalg.norm(v)
        
        # Find first orthogonal vector
        if np.abs(v[0]) < np.abs(v[1]) and np.abs(v[0]) < np.abs(v[2]):
            # x component is smallest, create vector using x basis
            ortho1 = np.array([0, -v[2], v[1]])
        elif np.abs(v[1]) < np.abs(v[2]):
            # y component is smallest, create vector using y basis
            ortho1 = np.array([-v[2], 0, v[0]])
        else:
            # z component is smallest, create vector using z basis
            ortho1 = np.array([-v[1], v[0], 0])
        
        # Normalize
        ortho1 = ortho1 / np.linalg.norm(ortho1)
        
        # Find second orthogonal vector using cross product
        ortho2 = np.cross(v, ortho1)
        ortho2 = ortho2 / np.linalg.norm(ortho2)
        
        return ortho1, ortho2
