"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi32>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi32>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi32>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, -2, 2], [-2, 0, -4], [-1, 2, 2], [1, 0, 5]]> : tensor<4x3xi32>}> : () -> tensor<4x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[(5.94600281869117,0.86775521945853185), (1.0960862804538838,-3.5040876535195684), (0.063281084451759645,-0.46486191243139552), (3.5678471282468438,3.0990927856634003), (-1.8802277458020162,2.976133697670071), (-2.6564495670447634,1.5968353027323903)], [(1.5956357130343652,-5.5250709219034526), (-2.377085038189394,-0.62118447583251379), (3.3002022618902167,1.8004754804893524), (-1.8025804413586219,1.862281665610271), (0.20806517220524118,1.6704238019865381), (-1.8760310808329961,2.960222373280557)], [(-3.0070052096542872,-3.0711417558300207), (-2.1281845124734229,-0.44472585396313635), (-1.6632141381101371,-1.6981827168304848), (8.135348322284063,-4.3371886833674775), (3.206588453295792,3.4852145833792632), (-3.282033380898647,1.1998585226472611)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xi32>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(2.6867237920050346,6.643368771063928), (2.6899736123397098,-6.6552580633003817), (-9.8002706310971881,-7.9270402195024658), (27.011551783779058,-6.200755126628696), (2.2365910705770693,9.5818489581255921), (-8.1249037342208296,-0.3270570958018113)], [(0.13601520123480881,10.549056584403019), (6.320565488985924,8.7870787228916818), (6.5262943835370288,7.7224546921847299), (-39.677087545629938,11.15056916214311), (-9.0658983215791356,-19.893125728857193), (18.441032657684115,-7.993104696053825)], [(-8.7687418119310134,-18.060180574925479), (-10.106625381779518,1.3722669939282681), (3.2106951631084,0.66944743974913035), (9.0976886336040383,-8.0489068211778125), (8.7095349968040825,7.3351430730615315), (-7.6596793564185228,6.7233264891232469)], [(-9.0890232295802669,-14.487953559691571), (-9.5448362819132306,-5.7277169233352501), (-8.2527896060989256,-8.9557754965838185), (44.244588739667158,-18.586850631173988), (14.152714520676945,20.402206614566389), (-19.066616471537998,7.5961279159686956)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

