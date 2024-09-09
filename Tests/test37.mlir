"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> tensor<i1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %2 = "stablehlo.reduce"(%0, %1) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i1>):
      %3 = "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%3) : (tensor<i1>) -> ()
    }) : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
    "func.return"(%2) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

