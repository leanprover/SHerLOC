"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf32>
    %6 = "stablehlo.clamp"(%4#0, %4#1, %4#2) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    "func.return"(%6) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-0.890106558, -2.546803, 3.41218138], [-2.42367935, 0.556205094, -4.37419939]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.682396054, -5.85568476, 5.77086878], [-4.9932375, -2.85505986, -4.2979579]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %3 = "stablehlo.constant"() <{value = dense<[[-2.02943087, 1.23262703, -2.46976185], [4.53440619, 2.10693359, -3.85020328]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1, %2, %3) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.02943087, -2.546803, -2.46976185], [-2.42367935, 0.556205094, -4.2979579]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%0) : (tensor<2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

