"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x3xf32>
    %5 = "stablehlo.broadcast_in_dim"(%3#1) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %6 = "stablehlo.divide"(%3#0, %5) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<2x4x3xf32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    "func.return"(%6) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-0.330817193, -3.21640301, 1.15269339], [5.66107893, 1.98200059, 3.72667956], [1.36263978, 3.81335855, 0.327093095], [-2.62778568, -2.15805292, -1.09849858]], [[-1.18254316, -2.92176151, -2.32717824], [-1.72833169, 4.70706177, 1.16282248], [3.48577952, 2.32715058, 0.348840296], [0.373283356, 3.3044579, -2.34404302]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[3.98198795, -2.23650098, -2.21360612]], [[1.605560e+00, -3.02210736, 1.74050951]]]> : tensor<2x1x3xf32>}> : () -> tensor<2x1x3xf32>
    "func.return"(%1, %2) : (tensor<2x4x3xf32>, tensor<2x1x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-8.307840e-02, 1.43814063, -0.520731032], [1.42167151, -0.88620603, -1.68353331], [0.342200875, -1.70505559, -0.147764817], [-6.599180e-01, 0.964923739, 0.496248454]], [[-0.736530066, 0.96679604, -1.33706725], [-1.07646656, -1.55754292, 0.668093144], [2.17106771, -0.7700423, 0.200424239], [0.23249419, -1.09342837, -1.34675682]]]> : tensor<2x4x3xf32>}> : () -> tensor<2x4x3xf32>
    "func.return"(%0) : (tensor<2x4x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

