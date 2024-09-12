"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6xi16>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi16>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6xi16>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<i16>}> : () -> tensor<i16>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x6xi16>, tensor<i16>) -> tensor<6xi16>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi16>, tensor<6xi16>) -> ()
    "func.return"(%6) : (tensor<6xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 2, -1, 0, 0, 5], [0, -2, 0, -2, 0, 1], [-5, -3, 2, 5, -1, -6], [0, 3, 0, 0, -3, -2]]> : tensor<4x6xi16>}> : () -> tensor<4x6xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 3, -2, 2, 1, 4], [2, 0, 0, 0, 0, 3], [1, 0, 5, 1, 0, 2], [-2, 0, 10, 1, 0, -2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 3, 4, 6, -1, 1]> : tensor<6xi16>}> : () -> tensor<6xi16>
    "func.return"(%0) : (tensor<6xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

