"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<i8>, tensor<2x3xi8>, tensor<i8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xi8>
    %6 = "stablehlo.broadcast_in_dim"(%4#0) <{broadcast_dimensions = array<i64>}> : (tensor<i8>) -> tensor<2x3xi8>
    %7 = "stablehlo.broadcast_in_dim"(%4#2) <{broadcast_dimensions = array<i64>}> : (tensor<i8>) -> tensor<2x3xi8>
    %8 = "stablehlo.clamp"(%6, %4#1, %7) : (tensor<2x3xi8>, tensor<2x3xi8>, tensor<2x3xi8>) -> tensor<2x3xi8>
    "stablehlo.custom_call"(%8, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xi8>, tensor<2x3xi8>) -> ()
    "func.return"(%8) : (tensor<2x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<i8>, tensor<2x3xi8>, tensor<i8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, -1], [-4, -5, -3]]> : tensor<2x3xi8>}> : () -> tensor<2x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i8>}> : () -> tensor<i8>
    %3 = "stablehlo.constant"() <{value = dense<-2> : tensor<i8>}> : () -> tensor<i8>
    "func.return"(%2, %1, %3) : (tensor<i8>, tensor<2x3xi8>, tensor<i8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-2> : tensor<2x3xi8>}> : () -> tensor<2x3xi8>
    "func.return"(%0) : (tensor<2x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

