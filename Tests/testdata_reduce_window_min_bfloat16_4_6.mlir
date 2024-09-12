"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xbf16>
    %4 = "stablehlo.constant"() <{value = dense<0x7F80> : tensor<bf16>}> : () -> tensor<bf16>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<bf16>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    "func.return"(%6) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.609380e+00, 3.031250e+00, 2.968750e+00, 2.093750e+00, 1.304690e+00, -1.765630e+00], [-4.937500e+00, 5.250000e+00, 4.648440e-01, -8.359380e-01, -6.328130e-01, -1.777340e-01], [5.468750e+00, -1.742190e+00, 4.687500e+00, -1.175000e+01, 9.570310e-01, 6.906250e+00], [8.164060e-01, -2.406250e+00, 3.375000e+00, 2.921880e+00, -4.937500e+00, -2.859380e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%1) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.937500e+00, 4.648440e-01, -8.359380e-01, -8.359380e-01, -1.765630e+00], [-4.937500e+00, -1.742190e+00, -1.175000e+01, -1.175000e+01, -6.328130e-01], [-2.406250e+00, -2.406250e+00, -1.175000e+01, -1.175000e+01, -4.937500e+00]]> : tensor<3x5xbf16>}> : () -> tensor<3x5xbf16>
    "func.return"(%0) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

