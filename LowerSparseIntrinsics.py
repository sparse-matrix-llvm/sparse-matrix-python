import pprint
import numpy as np

class LowerSparseIntrinsics:
    def __init__(self, dense_matrix=None, num_rows=0, num_cols=0):
        self.shape_info = self.ShapeInfo(num_rows, num_cols)
        self.csr_matrix = self.CSRMatrix()
        if dense_matrix:
            self.dense_to_csr(dense_matrix)
    
    class ShapeInfo:
        def __init__(self, num_rows=0, num_cols=0):
            self.num_rows = num_rows
            self.num_cols = num_cols
    
    class CSRMatrix:
        def __init__(self):
            # these would all be small vectors just like in normal matrix
            self.values = []
            self.col_indices = []
            self.row_pointers = [0]
        
        def __repr__(self):
            return f"CSR_Matrix(name={self.values}, details={self.col_indices}, row_pointers{self.row_pointers})"

        def __eq__(self, other):
            return self.values == other.values and self.col_indices == other.col_indices and self.row_pointers == other.row_pointers
        
    
    def dense_to_csr(self, dense_matrix):
        for row in dense_matrix:
            for col_index, value in enumerate(row):
                if value != 0:
                    self.csr_matrix.values.append(value)
                    self.csr_matrix.col_indices.append(col_index)
            self.csr_matrix.row_pointers.append(len(self.csr_matrix.values))

    def csr_to_dense(self):
        num_rows = self.shape_info.num_rows
        num_cols = self.shape_info.num_cols
        dense_matrix = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    
        for row in range(num_rows):
            start_index = self.csr_matrix.row_pointers[row]
            end_index = self.csr_matrix.row_pointers[row + 1]
            for i in range(start_index, end_index):
                col = self.csr_matrix.col_indices[i]
                value = self.csr_matrix.values[i]
                dense_matrix[row][col] = value
    
        return dense_matrix

    def csr_to_flat_array(self):
        # we can get len(self.row_pointers) and len(self.values) from SmallVectors.size()
        flat_array = [len(self.csr_matrix.row_pointers), len(self.csr_matrix.values)] + self.csr_matrix.row_pointers + self.csr_matrix.col_indices + self.csr_matrix.values
        return flat_array
    
    def flat_array_to_csr(self,flat_array):
        len_row_pointers = flat_array[0]
        len_values_col_indices = flat_array[1]
        
        row_pointers = flat_array[2 : 2 + len_row_pointers]
        col_indices = flat_array[2 + len_row_pointers : 2 + len_row_pointers + len_values_col_indices]
        values = flat_array[2 + len_row_pointers + len_values_col_indices :]
        
        csr_matrix = self.CSRMatrix()
        csr_matrix.row_pointers = row_pointers
        csr_matrix.col_indices = col_indices
        csr_matrix.values = values
        
        return csr_matrix

    @staticmethod
    # row by row multiply from: "A Systematic Survey of General Sparse Matrix-Matrix Multiplication" section 3.2
    def csr_matrix_multiplication(sparse_a, sparse_b):
        A_values = sparse_a.csr_matrix.values
        A_col_indices = sparse_a.csr_matrix.col_indices
        A_row_pointers = sparse_a.csr_matrix.row_pointers

        B_values = sparse_b.csr_matrix.values
        B_col_indices = sparse_b.csr_matrix.col_indices
        B_row_pointers = sparse_b.csr_matrix.row_pointers

        C_values = []
        C_col_indices = []
        C_row_pointers = [0]
    
        for i in range(len(A_row_pointers) - 1):
            # Initialize a dictionary for the current row of C
            C_row = {}
            # Go through all non-zero elements of the i-th row of A
            for a_idx in range(A_row_pointers[i], A_row_pointers[i + 1]):
                a = A_values[a_idx]
                k = A_col_indices[a_idx]
                # Multiply the non-zero element a with the entire k-th row of B
                for b_idx in range(B_row_pointers[k], B_row_pointers[k + 1]):
                    b = B_values[b_idx]
                    j = B_col_indices[b_idx]
                    # Accumulate the product in the corresponding entry of C
                    C_row[j] = C_row.get(j, 0) + a * b
            # Add the non-zero entries of the current row of C to the CSR arrays
            for j, c in sorted(C_row.items()):
                if c != 0:
                    C_values.append(c)
                    C_col_indices.append(j)
            C_row_pointers.append(len(C_values))

        c_sparse = LowerSparseIntrinsics()
        c_sparse.csr_matrix.values = C_values
        c_sparse.csr_matrix.row_pointers = C_row_pointers
        c_sparse.csr_matrix.col_indices = C_col_indices
        c_sparse.shape_info.num_rows = sparse_a.shape_info.num_rows
        c_sparse.shape_info.num_cols = sparse_b.shape_info.num_cols
        return c_sparse
    
    
if __name__ == "__main__":
    dense_matrix = [
        [1, 0, 0, 0],
        [0, 2, 0, 3],
        [4, 0, 5, 0],
        [0, 0, 0, 6]
    ]

    csr_matrix = LowerSparseIntrinsics(dense_matrix, len(dense_matrix), len(dense_matrix[0]))
    flat_array = csr_matrix.csr_to_flat_array()
    new_csr_matrix = csr_matrix.flat_array_to_csr(flat_array)

    assert new_csr_matrix == csr_matrix.csr_matrix

    csr_matrix_2 = LowerSparseIntrinsics(dense_matrix, len(dense_matrix), len(dense_matrix[0]))
    res = csr_matrix.csr_matrix_multiplication(csr_matrix, csr_matrix_2)
    res_dense = res.csr_to_dense()

    print(res_dense)

    dense_a = np.array(dense_matrix)
    dense_b = np.array(dense_matrix)
    print(dense_a @ dense_b)
