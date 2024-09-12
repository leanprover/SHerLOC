"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x10xf32>, tensor<3x2xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf32>
    %5 = "stablehlo.gather"(%3#0, %3#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 5>}> : (tensor<3x10xf32>, tensor<3x2xi64>) -> tensor<3x5xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    "func.return"(%5) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x10xf32>, tensor<3x2xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.4103384, 4.73668623, 0.673182607, -0.61154145, 1.11993873, 0.331038088, 2.017760e+00, 2.24512839, 0.0639716461, 3.36236548], [-0.542979777, -2.56619191, -1.39182234, 1.95056581, 1.0568105, -1.56906319, -4.38390303, 2.2276547, 2.44901681, 3.73211145], [-0.878893256, -6.06845617, -4.15767241, -5.33732271, 2.93643975, -0.195474297, -1.49406374, -0.341617793, -5.95141411, -2.91100192]]> : tensor<3x10xf32>}> : () -> tensor<3x10xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0], [1, 8], [2, 0]]> : tensor<3x2xi64>}> : () -> tensor<3x2xi64>
    "func.return"(%1, %2) : (tensor<3x10xf32>, tensor<3x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.4103384, 4.73668623, 0.673182607, -0.61154145, 1.11993873], [-1.56906319, -4.38390303, 2.2276547, 2.44901681, 3.73211145], [-0.878893256, -6.06845617, -4.15767241, -5.33732271, 2.93643975]]> : tensor<3x5xf32>}> : () -> tensor<3x5xf32>
    "func.return"(%0) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

