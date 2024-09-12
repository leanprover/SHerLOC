"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi16>
    %4 = "stablehlo.constant"() <{value = dense<32767> : tensor<i16>}> : () -> tensor<i16>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i16>) -> tensor<i16>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<4x6xi16>, tensor<i16>) -> tensor<3x5xi16>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi16>, tensor<3x5xi16>) -> ()
    "func.return"(%6) : (tensor<3x5xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, -1, 4, 2], [-1, -1, 3, -4, -2, 2], [0, -2, 4, -3, 0, -1], [0, 0, 3, 0, 0, 2]]> : tensor<4x6xi16>}> : () -> tensor<4x6xi16>
    "func.return"(%1) : (tensor<4x6xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1, -1, -4, -4, -2], [-2, -2, -4, -4, -2], [-2, -2, -3, -3, -1]]> : tensor<3x5xi16>}> : () -> tensor<3x5xi16>
    "func.return"(%0) : (tensor<3x5xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

