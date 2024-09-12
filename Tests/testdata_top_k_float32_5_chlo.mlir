"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5xf32>, tensor<5xi32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<5xf32>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5xf32>, tensor<5xi32>)
    %5:2 = "chlo.top_k"(%3) <{k = 5 : i64}> : (tensor<5xf32>) -> (tensor<5xf32>, tensor<5xi32>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5xf32>, tensor<5xf32>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5xi32>, tensor<5xi32>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<5xf32>, tensor<5xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[0x7F800000, 0x7FC00000, 0xFFC00000, 0xFF800000, 3.000000e+00]> : tensor<5xf32>}> : () -> tensor<5xf32>
    "func.return"(%2) : (tensor<5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5xf32>, tensor<5xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0x7FC00000, 0x7F800000, 3.000000e+00, 0xFF800000, 0xFFC00000]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %1 = "stablehlo.constant"() <{value = dense<[1, 0, 4, 3, 2]> : tensor<5xi32>}> : () -> tensor<5xi32>
    "func.return"(%0, %1) : (tensor<5xf32>, tensor<5xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

