"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xi1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2xf32>, tensor<3x2xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xi1>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x2xf32>) -> tensor<3x2xf32>
    %6 = "stablehlo.compare"(%5, %3#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xi1>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x2xi1>, tensor<3x2xi1>) -> ()
    "func.return"(%6) : (tensor<3x2xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2xf32>, tensor<3x2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.39667344, -1.37205708]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.23060751, -0.833974123], [-1.83391356, 0.285022348], [-5.98066997, 1.44504774]]> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
    "func.return"(%1, %2) : (tensor<1x2xf32>, tensor<3x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[true, true], [false, true], [false, true]]> : tensor<3x2xi1>}> : () -> tensor<3x2xi1>
    "func.return"(%0) : (tensor<3x2xi1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

