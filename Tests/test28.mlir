"builtin.module"() <{sym_name = "jit_f"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = "stablehlo.constant"() <{value = dense<[0, 1, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %2 = "stablehlo.convert"(%0) : (tensor<3xi32>) -> tensor<3xf32>
    %3 = "stablehlo.convert"(%1) : (tensor<3xi32>) -> tensor<3xf32>
    %4 = "stablehlo.divide"(%2, %3) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    "func.return"(%4) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

