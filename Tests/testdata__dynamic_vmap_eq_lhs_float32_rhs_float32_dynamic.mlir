"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>):
    %0 = "stablehlo.compare"(%arg1, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xi1>
    "func.return"(%0) : (tensor<?xi1>) -> ()
  }) : () -> ()
}) : () -> ()

