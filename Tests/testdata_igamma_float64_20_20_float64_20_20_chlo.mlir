"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf64>, tensor<20x20xf64>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf64>
    %7 = "stablehlo.compare"(%5#0, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %8 = "stablehlo.compare"(%5#1, %5#1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %9 = "stablehlo.or"(%7, %8) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %10 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %12 = "stablehlo.compare"(%5#1, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %13 = "stablehlo.constant"() <{value = dense<0x7FF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %14 = "stablehlo.broadcast_in_dim"(%13) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %15 = "stablehlo.compare"(%5#1, %14) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %18 = "stablehlo.compare"(%5#1, %17) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %19 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %21 = "stablehlo.compare"(%5#0, %20) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %22 = "stablehlo.or"(%18, %21) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %23 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %24 = "stablehlo.broadcast_in_dim"(%23) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %25 = "stablehlo.compare"(%5#1, %24) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %26 = "stablehlo.compare"(%5#1, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %27 = "stablehlo.and"(%25, %26) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %28 = "stablehlo.log"(%5#1) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %29 = "stablehlo.multiply"(%5#0, %28) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %30 = "stablehlo.subtract"(%29, %5#1) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %31 = "chlo.lgamma"(%5#0) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %32 = "stablehlo.subtract"(%30, %31) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %33 = "stablehlo.constant"() <{value = dense<1.7976931348623157E+308> : tensor<f64>}> : () -> tensor<f64>
    %34 = "stablehlo.log"(%33) : (tensor<f64>) -> tensor<f64>
    %35 = "stablehlo.negate"(%34) : (tensor<f64>) -> tensor<f64>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %37 = "stablehlo.compare"(%32, %36) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %38 = "stablehlo.exponential"(%32) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %39 = "stablehlo.or"(%12, %22) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %40 = "stablehlo.or"(%39, %37) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %41 = "stablehlo.or"(%40, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %42 = "stablehlo.not"(%41) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %43 = "stablehlo.and"(%42, %27) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %44 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %45 = "stablehlo.broadcast_in_dim"(%44) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %46 = "stablehlo.subtract"(%45, %5#0) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %47 = "stablehlo.add"(%5#1, %46) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %48 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %49 = "stablehlo.broadcast_in_dim"(%48) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %50 = "stablehlo.add"(%47, %49) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %51 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %52 = "stablehlo.broadcast_in_dim"(%51) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %55 = "stablehlo.add"(%5#1, %54) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %56 = "stablehlo.multiply"(%50, %5#1) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %57 = "stablehlo.divide"(%55, %56) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %58 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %59 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %60 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %66 = "stablehlo.negate"(%5#1) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    %67 = "stablehlo.multiply"(%57, %66) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %68 = "stablehlo.subtract"(%65, %67) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %69 = "stablehlo.divide"(%68, %56) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %70 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %71:15 = "stablehlo.while"(%43, %57, %59, %46, %50, %70, %55, %56, %52, %5#1, %61, %63, %65, %66, %69) ({
    ^bb0(%arg32: tensor<20x20xi1>, %arg33: tensor<20x20xf64>, %arg34: tensor<20x20xf64>, %arg35: tensor<20x20xf64>, %arg36: tensor<20x20xf64>, %arg37: tensor<f64>, %arg38: tensor<20x20xf64>, %arg39: tensor<20x20xf64>, %arg40: tensor<20x20xf64>, %arg41: tensor<20x20xf64>, %arg42: tensor<20x20xf64>, %arg43: tensor<20x20xf64>, %arg44: tensor<20x20xf64>, %arg45: tensor<20x20xf64>, %arg46: tensor<20x20xf64>):
      %231 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f64>}> : () -> tensor<f64>
      %232 = "stablehlo.compare"(%arg37, %231) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %233 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %234 = "stablehlo.reduce"(%arg32, %233) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %236 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%236) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %235 = "stablehlo.and"(%232, %234) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%235) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<20x20xi1>, %arg18: tensor<20x20xf64>, %arg19: tensor<20x20xf64>, %arg20: tensor<20x20xf64>, %arg21: tensor<20x20xf64>, %arg22: tensor<f64>, %arg23: tensor<20x20xf64>, %arg24: tensor<20x20xf64>, %arg25: tensor<20x20xf64>, %arg26: tensor<20x20xf64>, %arg27: tensor<20x20xf64>, %arg28: tensor<20x20xf64>, %arg29: tensor<20x20xf64>, %arg30: tensor<20x20xf64>, %arg31: tensor<20x20xf64>):
      %127 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %128 = "stablehlo.add"(%arg22, %127) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      %129 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %130 = "stablehlo.broadcast_in_dim"(%129) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %131 = "stablehlo.add"(%arg20, %130) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %132 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %134 = "stablehlo.add"(%arg21, %133) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %135 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %136 = "stablehlo.multiply"(%131, %135) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %137 = "stablehlo.multiply"(%arg23, %134) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %138 = "stablehlo.multiply"(%arg25, %136) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %139 = "stablehlo.subtract"(%137, %138) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %140 = "stablehlo.multiply"(%arg24, %134) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %141 = "stablehlo.multiply"(%arg26, %136) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %142 = "stablehlo.subtract"(%140, %141) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %143 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %145 = "stablehlo.compare"(%142, %144) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %146 = "stablehlo.divide"(%139, %142) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %147 = "stablehlo.subtract"(%arg18, %146) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %148 = "stablehlo.divide"(%147, %146) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %149 = "stablehlo.abs"(%148) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %150 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %151 = "stablehlo.broadcast_in_dim"(%150) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %152 = "stablehlo.select"(%145, %149, %151) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %153 = "stablehlo.select"(%145, %146, %arg18) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %154 = "stablehlo.multiply"(%arg29, %134) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %155 = "stablehlo.subtract"(%154, %arg23) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %156 = "stablehlo.multiply"(%arg27, %136) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %157 = "stablehlo.subtract"(%155, %156) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %158 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %159 = "stablehlo.multiply"(%arg25, %158) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %160 = "stablehlo.add"(%157, %159) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %161 = "stablehlo.multiply"(%arg30, %134) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %162 = "stablehlo.subtract"(%161, %arg24) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %163 = "stablehlo.multiply"(%arg28, %136) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %164 = "stablehlo.subtract"(%162, %163) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %165 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %166 = "stablehlo.multiply"(%arg26, %165) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %167 = "stablehlo.add"(%164, %166) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %168 = "stablehlo.multiply"(%153, %167) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %169 = "stablehlo.subtract"(%160, %168) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %170 = "stablehlo.divide"(%169, %142) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %171 = "stablehlo.select"(%145, %170, %arg31) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %172 = "stablehlo.subtract"(%171, %arg31) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %173 = "stablehlo.abs"(%172) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %174 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %175 = "stablehlo.broadcast_in_dim"(%174) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %176 = "stablehlo.select"(%145, %173, %175) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %177 = "stablehlo.abs"(%139) : (tensor<20x20xf64>) -> tensor<20x20xf64>
      %178 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %179 = "func.call"(%178) <{callee = @integer_pow}> : (tensor<f64>) -> tensor<f64>
      %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %181 = "stablehlo.compare"(%177, %180) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %182 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %183 = "stablehlo.broadcast_in_dim"(%182) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %184 = "stablehlo.multiply"(%arg23, %183) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %185 = "stablehlo.select"(%181, %184, %arg23) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %186 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %187 = "stablehlo.broadcast_in_dim"(%186) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %188 = "stablehlo.multiply"(%139, %187) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %189 = "stablehlo.select"(%181, %188, %139) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %190 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %191 = "stablehlo.broadcast_in_dim"(%190) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %192 = "stablehlo.multiply"(%arg24, %191) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %193 = "stablehlo.select"(%181, %192, %arg24) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %194 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %195 = "stablehlo.broadcast_in_dim"(%194) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %196 = "stablehlo.multiply"(%142, %195) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %197 = "stablehlo.select"(%181, %196, %142) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %198 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %200 = "stablehlo.multiply"(%arg29, %199) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %201 = "stablehlo.select"(%181, %200, %arg29) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %202 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %203 = "stablehlo.broadcast_in_dim"(%202) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %204 = "stablehlo.multiply"(%arg30, %203) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %205 = "stablehlo.select"(%181, %204, %arg30) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %206 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %207 = "stablehlo.broadcast_in_dim"(%206) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %208 = "stablehlo.multiply"(%160, %207) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %209 = "stablehlo.select"(%181, %208, %160) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %210 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %211 = "stablehlo.broadcast_in_dim"(%210) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %212 = "stablehlo.multiply"(%167, %211) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %213 = "stablehlo.select"(%181, %212, %167) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %214 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %215 = "stablehlo.broadcast_in_dim"(%214) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %216 = "stablehlo.compare"(%152, %215) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %217 = "stablehlo.and"(%arg17, %216) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %218 = "stablehlo.select"(%arg17, %153, %arg18) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %219 = "stablehlo.select"(%arg17, %152, %arg19) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %220 = "stablehlo.select"(%arg17, %131, %arg20) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %221 = "stablehlo.select"(%arg17, %134, %arg21) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %222 = "stablehlo.select"(%arg17, %189, %arg23) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %223 = "stablehlo.select"(%arg17, %197, %arg24) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %224 = "stablehlo.select"(%arg17, %185, %arg25) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %225 = "stablehlo.select"(%arg17, %193, %arg26) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %226 = "stablehlo.select"(%arg17, %201, %arg27) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %227 = "stablehlo.select"(%arg17, %205, %arg28) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %228 = "stablehlo.select"(%arg17, %209, %arg29) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %229 = "stablehlo.select"(%arg17, %213, %arg30) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %230 = "stablehlo.select"(%arg17, %171, %arg31) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      "stablehlo.return"(%217, %218, %219, %220, %221, %128, %222, %223, %224, %225, %226, %227, %228, %229, %230) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<f64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>)
    %72 = "stablehlo.multiply"(%71#1, %38) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %73 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %74 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %75 = "stablehlo.subtract"(%74, %72) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %76 = "stablehlo.not"(%27) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %77 = "stablehlo.and"(%42, %76) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %78 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %79 = "stablehlo.broadcast_in_dim"(%78) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %80 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %81 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %82 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %86:7 = "stablehlo.while"(%77, %5#0, %79, %81, %5#1, %83, %85) ({
    ^bb0(%arg8: tensor<20x20xi1>, %arg9: tensor<20x20xf64>, %arg10: tensor<20x20xf64>, %arg11: tensor<20x20xf64>, %arg12: tensor<20x20xf64>, %arg13: tensor<20x20xf64>, %arg14: tensor<20x20xf64>):
      %124 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %125 = "stablehlo.reduce"(%arg8, %124) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg15: tensor<i1>, %arg16: tensor<i1>):
        %126 = "stablehlo.or"(%arg15, %arg16) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%126) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%125) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf64>, %arg3: tensor<20x20xf64>, %arg4: tensor<20x20xf64>, %arg5: tensor<20x20xf64>, %arg6: tensor<20x20xf64>, %arg7: tensor<20x20xf64>):
      %100 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %101 = "stablehlo.broadcast_in_dim"(%100) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %102 = "stablehlo.add"(%arg2, %101) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %103 = "stablehlo.divide"(%arg5, %102) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %104 = "stablehlo.multiply"(%arg6, %103) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %105 = "stablehlo.multiply"(%arg3, %arg5) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %106 = "stablehlo.multiply"(%102, %102) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %107 = "stablehlo.divide"(%105, %106) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %108 = "stablehlo.subtract"(%104, %107) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %109 = "stablehlo.add"(%arg7, %108) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %110 = "stablehlo.divide"(%arg5, %102) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %111 = "stablehlo.multiply"(%arg3, %110) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %112 = "stablehlo.add"(%arg4, %111) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %113 = "stablehlo.divide"(%111, %112) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %114 = "stablehlo.constant"() <{value = dense<2.2204460492503131E-16> : tensor<f64>}> : () -> tensor<f64>
      %115 = "stablehlo.broadcast_in_dim"(%114) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
      %116 = "stablehlo.compare"(%113, %115) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
      %117 = "stablehlo.and"(%arg1, %116) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %118 = "stablehlo.select"(%arg1, %102, %arg2) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %119 = "stablehlo.select"(%arg1, %111, %arg3) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %120 = "stablehlo.select"(%arg1, %112, %arg4) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %121 = "stablehlo.select"(%arg1, %arg5, %arg5) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %122 = "stablehlo.select"(%arg1, %108, %arg6) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      %123 = "stablehlo.select"(%arg1, %109, %arg7) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
      "stablehlo.return"(%117, %118, %119, %120, %121, %122, %123) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>) -> (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>, tensor<20x20xf64>)
    %87 = "stablehlo.multiply"(%86#3, %38) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %88 = "stablehlo.divide"(%87, %5#0) : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %89 = "stablehlo.select"(%27, %75, %88) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %90 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %92 = "stablehlo.select"(%12, %91, %89) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %93 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %95 = "stablehlo.select"(%15, %94, %92) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    %96 = "stablehlo.or"(%22, %9) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %97 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
    %98 = "stablehlo.broadcast_in_dim"(%97) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<20x20xf64>
    %99 = "stablehlo.select"(%96, %98, %95) : (tensor<20x20xi1>, tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xf64>
    "stablehlo.custom_call"(%99, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    "func.return"(%99) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf64>, tensor<20x20xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x707D09860071FB3FCBD725D72BA3C13FA816AC2C5216E6BFEEC92256BD74EDBF9926DC56387BF0BF9EC50FF1C5070140385F6B96731FFE3F920DC3B918DBFE3F1A8429454EFB18C0786F8FF8B38AF6BF645B4EE42F7AF53FE2E9BAB07AB9F73F9152A9F1CD960140E00ACC45BB901840BC71C44FFA26FF3FE949F3E6166002C0E4086D1E5465E9BF4AFE4745B8E5E13F4A3C5DB83A7E03C05808B8623EFB0EC01E11D0071EA416C0AC6AAB7C84BADEBFC366809825B90640A2343909E43113C0A98BDB5348D5F3BFBBC06AC7411D02405BA269062FC3F2BF9314CB6C0F49DF3F7BD7D92FB7B11440347BB677DFAFE33FF6CB138FE21018409E46899BEFF60A40CCD58907E3F4004048D74B2A8B89EDBF049EFB3D7BA9F2BFD4558BDDE84ED3BF5DE3EB490AEF0DC0BBA1C4212E3310C07C6B1A37079613402375B1FCFF4301C0CAB408B30CD9F73F36BFF5BC88690540F4CA84FFADC9F2BFAE4C6871AF640240B6696F2110D705C0EEBF870752F21C406D89ABBFA7CDC0BF6648AA677E27F1BF9A73694CEC0AFF3F886E5DBC939FECBF068B6AAD81820A40266A8D5CA6A4104072C69ED513CCFE3F30B4D75CC23ACEBFDCCAD1F34663F73FA0C773576021EC3FE4743CA39B11E83FF0DC97AD582F064054DFA0DC59F0EC3FC0BB492487121540C69C3E7BE17D0AC08B2E825A9C41FF3F2611FD59783303C0E2A671951CE90240A26B87119FEC05C02AAEF818C9C10A401975233DB5070440A4977148872F09C064D06C8F56DF084064E886A20D1CDB3F6D36FD266B4A09405BFB10666AAA02C0D8B44907872CF5BFE6B2D513ADAB0FC03BF5BC2B21E105C01A049E9A9BEF16C034B84D3CA484C93F593041C4B4CE00C0F795217E94FFF5BF7410937B880CF4BF76D11AC717E9D83FBF35ECFBF1630440CA906914716EE53F1BE47C235215FA3F2A4A3E876BEA1AC0373537B68F1AF33F9419C872D620FB3F875A23D7DEB0E83F77A87207BCFA21C00244E3763C1A14C0A0F7F522F2F5F0BF8AEDFB560A59F7BFF43D7E23BDF1FBBF4EF9C65819CFCBBF8CA9E0019F5E0840AC5D9C221B6C1140508F0434395D0540CCDA43526C1212C0548B097A3E23C23F2E483B0A3041F13FBE013DA4677ECB3F6D6F29015F9A1DC0E96CF10C7C2A04C0C6A0B0DFC28708C0CAE1376ACB71F63F30AC43DD82BF07C0CB55550ADAFDC1BFC01857BC991A134022724AF77DAEE2BF8D88FA28A797F83F794B6087DB53D33FEFA7741CF20496BF2855AB95DEAB0E405BAE80772EEAFBBF3E10C536C7DDFC3F089F812DC8DAF6BF3B710B9B9FA70A4052803E329CD2DCBFD8D9DD998223D43F86A584851A1602C02ED54436FA67014075C467452E2AF1BFC8CFAA6F61871AC0A0C1DBB975A9E83FA2DCAE9FFD3EF9BFC50097CE4BFDE4BF9882ED5FE85200408C32D5B5E09407C06ED3BCA495010640148ADCBDFCF804401E23388F57D71340EF314905045505407C7BA75D63E118403252F951905804C094D2B70A9501F7BFC391361A15891040A8A064F800A5D7BF1147ACDD348C03C0DA44AAA4A7F31440E170B8E02A1902404F849521CF12F7BF92494F4C62DEE3BF6240604C21311540F986D197DA4AF13FFC306FB1D10002C0D8FB86D11482F83F9647BA7F7314D83FD30CF4926AE31440263AD52F193D16C006A3187A7598EB3FBB12862E05120BC00CEA249B628914C0180A88831B69DFBFE2552E3BE0DC05C0B2511FF7D5BFE13F54B26F12BAEBE33F1384EA29171814403D13993DB88704402FE48F3C611304402B4D41A56F6D08C00040AA0489AC19C02E519CF8997109C0183FDD8D3572DABF9089024E734A0E40AF05A86BA2C514C0C23B61462AD4CBBFEA8A8E3E8D9EF83FC2D6556C3CE6DD3FF2685838B062D73F1A80A2B704061540DD8F605BEC49D3BFE5BE9DCE89720AC0B4CEB1223EB21E4048323770F495FABFAE2DF1448F0903C0CEA29A55D12DE4BFC78BE8824CA7034066F9F95BEA1AE4BFBAEA2E75F79E23C01A186E6AB0FFF43F091DEF824A21064054BD621E9FE4F43F12790AF0A998EC3F853F3F9BBB5519C08E67F5090E3FE03F642713C9A415FABFDC16CD40BA5ADABF494A6095173B0840EF6E8BBE76B3EE3FB8EB22A5C7740EC052A10BF110480E40B805884EBAB300C03677098E1337F8BF7A55AC74827002C0D5143B87F2D5E33F9FBA687A195CE63F625308484882EBBFBAED6743E31F0840D4CE0C75C1A209409219B87851E501C040E1D405D7C91BC0021EBA84EEFE08C0644D9A1662F300C05A85B695FDA70EC0B4BF8E60480104401ABBD0E23985EFBF5395A6AE053199BF56043D450B4305401E7750396CF10BC0C050920E29850CC02ED88660B99D17C05563BB482DC1C5BF4D2A43686AC51C4020BF4E179CFAF83FAA929AE8E31D09C04981F3E4839308C01AFB7307AB4F0240D2FC2794784F09407C8BC07594DDF23FFE30611F507AECBF777ED2670E7907C05CB016006B9806C0287FCE8A7F77F9BFC67B95FC2C90FF3F8D29497A0E7E04C0CE0AFDA7707A19C022FB2B83126F12405154EFC2B2A2AFBFB49E4A18BD301EC0A604E1C1440D03406E796FC94F4AEE3F7C74DE53A1951B4073DBC7276A1F00407FCC58D15520F43F7BAF69953F79124011EDD14A0263A13FE3DCBB80079C06404E382FB4563517406A70051C41F90A406DF6095DB02007C06749AC91AC82D4BFCA0CB4A9AB80B43FFCD97260C495F33F86BB26B68DD5BABFF1668F729B4615C085D56838FDE505C09616C03B2841104036CC24F0DAE4E5BFE5AC6CDFAE1B12C0E402A2E9877AF1BF76195AE065821340EEA253D2750BF53F2DF7204781D517C0740749E883721040FBAE89C20FECE1BF9E143B6BD550EFBFE71BA97BD8E80C40918E24D8497410C002B927EE76CAEBBF1AB733ED1A8DC43FA23237C1A9C512C08E0660C3C1FDFD3FDEE65AD0171813409CDFE85C71F61B40CF5872784205F5BFC6669190BF8C00C0C9E1D4E1DBCFF4BF4EC53F90930215404F1E6621C1C000C0BEDB3713EC8EE83F5A16FECD71300D4024D0CA336EE914C0423DBDA63172F33F5617370F8E34FEBF58419A99554718C07C1F27FFB3D20F40A4C614DDBFF810401600B86A33D4F93FAF3256E2839DDEBFC18EF4A9245805C028B87BF22F090940D6046AC5A1E4F0BF9D7C78138543A73FD664061A3DF612407461599F2E2B15C02E84778D9ACD1AC0AA4E0FC853B408C00474EF4B20D0E33FB00A49F111431DC00A0BBB63D299EBBF5D13AECBFE7507C0F54C563F049B0CC0361CCF1587A4CCBFE21AD0024CAE12C0D08540515457F7BFA003A30F5B230BC0820C77F851F7FFBFF0D8AABD965712C0E430FAFFCD9A02C094354634A3F7094001409F44D6F0FFBF18E3854C148600C041F5021B740F05C01A87803E76CA124026C8E6550227C8BF68C66CD97F000DC0748120366992F13F45F6F321E0BDF6BF102C82056521E1BF163967CDC2721940577810F3CE0516405DB116C6E9BAF1BFA4DF7BA913AF793F0083E7477099C13FF07F9D96F6C3DF3F9E38C5C64642FABFA487BDD9647AD5BF5A9F681C10B41340354498E5C146F33FBE134A72111AFA3FA673DD38D41D02409E544F5BAC4BF5BF5569BFC9914CD43F55F1A124D07DEA3F7050B7701B510E40CF1C10FE196BDDBF1DB27BAD90E20E405685D083C51911C042FA7BA9B70DF4BF02A7CD2EF680EF3FD9C65A1126AA0440F46383790F90F83F07D33FE08C6E06C01D08BA860776ECBF9A11B499F0F5F1BF8F0139B0CB6E02C07D10E4FCDF4E75BF44A4D5C21211CA3FF65579719DEC06C0B7B3E69700801D404A6B666B4E4E01C0DC4D91352D801140966832DF332CF5BF5DCF7F45D686F2BF9ADCDB7661790840642F64C7C5EB16408DB9EEDDEDB20540EBA621F4E8B806C003E2B9EC0BE2E73FD749DC8FABFD07C0FAAB541BE3F3B0BFC7029FCC238F14C0C4D52F46E6CBE33FA716255707180940342DE925367815407E549758279513C05EC81EFC5350E2BF7E3AF6BBBC190AC01348A7179D15F03FD4B38A0E07DDC6BF6DD0438C68750DC06CEC1F13482AF9BFA0292FA329611FC0B41EAE27483FF03F4F6CDAC28A1E14C0896DB73E6E9D10402460A9316EB4FEBFD06C538C65E1F63FF843BA518F9D04C01185E225F7B7F43FE6B2523B02F7D4BF3BD32A19320C13406761432B17B8F1BFC5C58052FB7AFF3F730C535892470640485519A132941540D6180BD6C385FDBF3593A18730ACEC3F8DB6AFF9844411C0A1A391FA045F02C08A8307C7E9F301C0C5CFC35947F1F13F942BD6861ECDF8BF5324B054787220C05F5F3A69DB99D53F680EC7E27CF000C04CF44E62E25EDE3FB7ECE7AD2D7E0140D636702E3619B23F675002A2EFB8D73FCAFDD9C007D912408ECF08AB54AE004094809FA118CD13C0C20B2CFC8F5CF9BFF8DD78283495FFBF589420DA5FE91240926A8A212096F1BF066037E1E615F13F6C31904F9D55C6BFFAEAE27573B91840"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    %4 = "stablehlo.constant"() <{value = dense<"0xD9167815DBB7C93FFC6C47577B04F23FA6123A9943D6ECBF7C72B11C065C1640AB46378A0D031B405D878D15761BB5BF12DB84F802E603C0BB8C5C17CC3C14C0C0229AF0058A0B406F6EBB1F2DB5F4BFFE44B259E8F7E8BF58A07070BB790D40A16CF5D04452074078789C3D14BDEB3F1D64BA4644C908C056C9EAEA099DE13F1D11E6021EB51040F4142B7081600440965CAA3BAF5ACD3F80FC86931833F1BF858557A361C01340108189B2951700C0518AFE1F2994F5BF3CB1B5B2319CC23FAEA53DCE15790A401C4061E9C16C02C0029FD0C14E08E9BFDC053F1570FBF03FB73E506C1CD00840A147ED24517CFF3F80CC024A948AF93F240EBED6761AFC3F7AF19126575EF83FE04BA4A84850FBBFA1837BA42D9BCFBF88CFB2D88E220C40327637327387FD3FE539897DEAF0C5BF6806A6A51ADAFDBFEEDC86C2CEA9EFBF4A65D7456CEA0340B9B3A9E7F0F8ED3F08F5BCC1456006C0FC924CE7C2EAFF3F615CB683BA2DF13F20CF94667A0810C05DC3FDEC101FEF3F97E8D7EEA9A0E13FF1B91901D8E001C036D7E926D463F93F85A73160EB1BF1BFD94FE3C3D4D1F7BF370255AC8B2AEB3FBC9E097496A0EBBFB11F111A95FA0340A8121ACD6EF01A40E7B1268EDAB804408EDF991422360740F231B11BC051EEBF8A30B518139AECBFCDAA2A4CFC370FC0EF1543FC9E1BD4BFF95F9747E7230AC0B06B79B650FBDABF574BE42FEDE5F53F48D0D3CD4CC6A8BF2EE54384E21A03409FBD65973CBA00C00E7ECBB7FBB1D0BF063CDA094DEA13C024DDA89FEB6FE7BF40F872EED020FFBFE6E3A3847D31E6BF11A0E8A3444312C0ECDF5CD1C22205404DD8E9CDC659D23F39E6E47F8E5E15C01E993D104D0B0840D203CC9ABAE8FE3FAEDE0A99A7A3F93FAE5833FA80E71640672486A3E4360140C6BDE63300FCCA3FD78B563F5AD8F43F461CA8337DFA0EC0F8D4BA84911B05C036B199F6240B1040864922B19E07F53FFAD9A227F7900CC04BF8A633D34BFABF56401A8A98910340E179C7E84F300AC00476A70EFFA6E63F3262DC85C60809C0DF0FFF81438FE23F1DEB0414C7880540086777838B2213C040CB682B5769DABFCA9570D5E06BF3BF448D853534D810C03AF86C56342301C05D4AF8F88E0EF03F199A213653ABF23FF0ED003B424505409612E55CAD2CCFBFD02B6CFD3C37F8BF5EFB6919A5AF19C03E4BE752E631F5BF643FFCE0F8B4F4BF9CE4011B1FBEE73F952F3AFC226FF83FB3D098BEC0961240F6AFA214EEDB58BFFB3B3257268B01402CC3990C4CD910C0AC5AA31EC2C8C23FCE446357D02D0BC052F2CD699E2F0FC0460977025FB7E03F14C35EF7E6B0E3BFE6C284E3511204C05686DF3851DA03407290678465000940D2008D8E50E1F1BF5D39738AA5C70440239BB6699125C9BFDD8579C960DE16C0548F8FBC9B2EB93FCC74808FAC38D3BFEEBBA377E3B3F43FBA44070CA98ADC3F501799EB9E69F4BF96DC5841395D13C04B4D389A6DCA02C0434726D3B62409403719BAD60A7FA93F9A4A6B1728830140CCF67F782F4FF73F9402218F573C06C02921E2F5E3F1F9BF086D7B19F0C4FABF9C90A3B0320117C0DE3771562899FA3F467397E56ED5FFBFF2BDFEF289B7F93F4E16048C50A4ED3FE828E64AC763F5BF0401FE61F5B9EB3FC4A06BDE311BFC3F31078241BEEF0F403605A89DCEA706400D8B47C8380710C013C9F7EE490FF03F2219190D2F361AC061C68AB2C08FED3F8DB7850CB2A114C09CB91BE6ED3D154012435D628DDB1AC0EE9F68FEE524FDBF34AD0A673F10ED3F0B38387D4531FE3F2767D847B24BE43F813BED12C7AB1EC0189F85D5D8E200C0798E69CBA582F9BFD7B60B64D54613401DF99749B7AA00C0F664B670EBFAF03FE484B364149812406AC07C8403E6D8BFE423AD276723174001EE8988E46B12400F0241EB3548FDBFA17E3CCAE89BE63F6AEC9E392BC7F33F5838395D0887084074B371BBBEC516407ADCC52B41AB07409F144B2FE06D1740E3C5C5CEB136ECBF5CEBC4EA69FD0540A65368A2BFC90640717E9BB276DBD43FFAE097D6E32DFE3FF1872D131463114077F6BF2A9DCC06C0BB9FDD52AE681EC01AAE78C69B8CFDBF1C676AEE2DEA1340F0856CE7171E10C006D5BB3275B010C032E0948928A204C06CA2EBE22B7906407F26721051C31440A533496A708D1340B8A340CEC804F8BF1E525BE679B1DF3F57288308415F2040DA045A3CCA9DEBBFFC9C0E45F1540B40798C2178715000C07E41907D955BF6BF12301FE637E90140C2C9128170FFF9BF0888E4EC5AD019C04688914BDF4CFF3FC9F924FB7246F63FFF1D370DA078B5BF8038B7B2EA1B853FF65B55637801FF3F659CDC0FE934E4BF4062FFFC1964F43FCB2847FB9B38C1BF9135F8D8403E06400A7A08B09A6B0AC0C6B2C2C1485100C068BF3E1B039DFCBFBEC871D40C2110403A9C0EA916CCFA3FCF288E14FAFAF83F86C97EBDE056F73F22BA795FF0C20CC01FCB5FE5154BE0BF4A96275D85720640F4CD8356DD28B8BF55BFF41A8BB6F43F66F32007D5A80C40B663B899E7B50440A2115E740BB70040D89DE593CCFF04C0F21D80E54C6110C01971C4FB9094CDBFD3D8E7DB4333D2BF90EC2740FAEBFCBF78002265E58DED3F982124B71E6EF3BF86C0BEA60DE4993F0460B04212DB05404559F3C3EC94124052AA4DB749EE18C00245484CB9A5DEBF3649451E190503406CD9666D85CDE33F0252BA8E372812C04A62842E57A1D2BF82DD79BD6BF80740A67A8409DEDA114088489D97622C0FC0A17DFCEB58150B40E65FD7772173EEBF27248AD3D2EC16C0DC3E8CA59A6A1240E3A6634A0DA307C098E723F3130CD2BFC50D603D756C13C0687F20143020F63F44B84D524107DABF330298F7314F1440EFD2AE17877906400CA88415E7E60740B0CAA9086AAF1740D0C6659C3A2DEBBFBB7E42F7EB65F03F0B540A13758CDB3F961F3FCB5F450FC018578A3BB0F010C0265D1A1E232AFDBF08A4E564A29CB33F93D14A07BA7406C04CD8593972FF184060E31E18F7BDFC3F801BDFFFD8820040DAD70DE07F131140B04E790CA289FABF56943F97250004400241E1BA0EB603C025BDC4DC19340FC0566319775DF108C070B554AEF62AFABFB6237A0D1C480DC0275E50D28F27F0BF0E62FD4B955A1240E20E3447977E09C0B04146FD2EB9D43FA4E81E1E79741640B57AA0FDC6ACD33F86B5F030C6FBF5BF00D118662BDFFFBF26CDDBB12537D83FC458CF72835FF4BFE1E9D60B5C80F8BF34F8F3024869F13F1F907A126F200140F210A34631FD02C004CBF651B4CFFF3F7CD453F8E556E73FFE03B732BC7A0CC0FEEF60147E9BE23FAE38F1A7843FED3F35B53A252BEB10C0E98B0A7A4C2DFABF9A572D42A0D6C63F048B2A2798940EC04CA85C3098BF0940D19708AF84E4E73F6E16AF93D466F8BF0683463DDF9D00C03319C7D98B6207C018C435E8B653F13F5E26A217034413C0AEA51780ED6508C0A5949E9D9E89D43F7B3A71112FC90A40C65C3AC7E4F9EE3F460A027B6EC906C05F437705246F0340093AB67B2DFAE13F5D20C9B920930B401C7AE958C2FCFBBFD8635D608D92ED3FA6398BE290440F408125C8872B69BD3F3AD2BFCA3DD00B40B210F4384310FE3FD07BB114D534E63F19F281DF8FB900C09061D329807D1140ADEC4BB5042BBABF9DC9FF9FFAC708C09C543C06405115406DCE88544D6E094052FB5046A9E3224069AE08AE1F4FFD3F546B91AA65E900C0555F00EB14E10B4068ED7AED457D09C030200227934B00C01CAE051B955101C026B025AF3DC2FA3FA7BAE12C755FD03F92DDC8954B0B16C05F94F0BF2E0E0F4090FFB76D11BF0C409BC4479EACD3FB3F83A619CA9D7910400C2FAC752BD605403C5E7E803D6A174018BBB50B917503C0E63EA09755B7F5BF5CBD7798716D15407BF597E997FAFABF97BBD6A90CBDB7BF6F2B5B6F9DC3D33F7A201548B813D0BF628A6E0BBB1EFD3FAEE7A24B67C80E4052B57099117C00C0E6347D82AE0C0C40EAFBABF88324094030183383780711C0A649BBD0F44CF13F3C641D5533EDFEBF452C852C4F78FEBF9BD9451E393607C01397D31EB203F1BF302A075F79FDB13F7E546C55F8BF0CC0CECC17B83F32F8BF029A964ACC3BC0BF9C0AC15283A2F4BF9D39966E7CAF14404624F56B1C2ED8BFFE709706B59C07C052844D41639A134034C47B92611CF03F705E1B35318E0AC0AC6C10F45F06EC3F112E82492883FF3FB478C0F578180840A4BFC50C9286C33FD8578ECFA7ABC43F4C70B31C61480AC0F6E2ABEC613DF8BF228374197783FB3F0DACF97166A9E1BF3C470D4EEC1CEF3F70D07C308D2AE2BFA4ACAEA4A2E50B403A2023C6506603C0C9B84E3E8894094090F022E0A373AF3F54CAAB0645498C3F24E1EF6C7C38EEBFCF1AC2AD7A3912C0DA62958FA4600A407401C6AD60D20040E0D3843F2E4EF3BF3C23EDEA549FD73F7B21B081820A10C03E20C0F970250CC0"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%3, %4) : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x6C7D4A38226BA23F6EE8AB029218EF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F80AADC9CF418EE3F8E86BEABFCDAE73F97FBAFD93DAE2B3F000000000000F87F000000000000F87F000000000000F87F89DF38AE9514EF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F79D34E46787CEB3FEB44023D6DE4C63FD80150BF38EEED3F2CE46694AEE6773FDA3EE25447F2C73FD879F4AADC6FDA3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F9D249A6E4784EA3FBBF70867614CBB3F000000000000F87F79DD2CBEBB2CE03F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F173EBD7B4736CD3F000000000000F87F2401D7CE62BEEA3F8A36E3D2FDF8EF3F5BDCDCB6688EEE3F7F750568157DE33F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FF49CA5F23FC4E13F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FCFC8A7DF4FFCEF3F0C87DB697DBADE3F22D39838B2FFD63FBECBA6C736CCDF3F000000000000F87F000000000000F87F062AEC2C560CEE3FC17B743A76F4E93F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FA730D60E81EB933F2014B571157BCC3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F81399F050F45D33F72C05DF86BB0EE3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FB3AF7C5213F9E93F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FFDCCB959BEE5CA3F3650B61DEF6E1D3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F863B915EE209823E000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F6C643B76C119933F000000000000F87F000000000000F87F80EBBD04549FD83F000000000000F87FD27269693B6E553F000000000000F87FF1AF695C9D91EF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F1AEC30ABD3AEE93F000000000000F87F73D84EDD8075E33F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F7DC9E6DE3DBCEB3F9E1682821EF5EF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F95F3BD4B67A8EE3F000000000000F87F000000000000F87F000000000000F87FCDE1CC3FB36DE23FA5F1A6FF84E5EC3F6680A250E017D53F000000000000F87FC33CC9081BE5EF3F000000000000F87F000000000000F87F000000000000F87F8A29E339D2CCEF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F3C4D5CC917EAEF3F000000000000F87F000000000000F87F10768E73969CEF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F998F8C8F8F54EB3F000000000000F87F000000000000F87F000000000000F87F601B79F73696E73F9754AC553753E83F000000000000F87F000000000000F87F000000000000F87F000000000000F87FE310B1549ED3E83F000000000000F87F000000000000F87FE4C64E1B8F01D73F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F8FF4C465D424753F000000000000F87FA1F07BE8B307DA3E531A032AA53FB23F1D983032CC36E93F000000000000F87F000000000000F87F586B4A8212EAEF3F1EF9B287ACFFD63F000000000000F87F000000000000F87F000000000000F87FC0CF80683177E43F000000000000F87F000000000000F87F000000000000F87F000000000000F87F43BF22D7BE62EF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FE83FA690D3E8EF3F000000000000F87F000000000000F87F004E25FDAF60783FFA7BB8163EB9993E000000000000F87F000000000000F87F000000000000F87FD3CB8CADFDEA3D3E000000000000F87FF43F0C9CAEF7EF3F9796AA1CBD7CC33F000000000000F87F06ADDD34184DEF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FFA9D2B45A0B5083F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FA2AAA31D194BD13F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87FC22F8EEF8FFFEF3F4C178184C5D8EE3F000000000000F87F000000000000F87F000000000000F87F39ACA3046BB3D13F000000000000F87F46F18A0D644CD63FB7C9AB1C09C5EB3F000000000000F87FD75F97F704DFEF3F55008A13B65FEC3FA2CE28F8B8FA803F000000000000F87F4B8779A05231E53F000000000000F87F000000000000F87F379E08AFC1D9EF3FA6481C3B5EB1E63F1C5DA1D870FDEF3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F15382B13713D2B3E000000000000F87F04A8AA74CEDDDD3F000000000000F87F000000000000F87F91862AE17098E83F1EA44811586BB33F566953130164EE3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F55EC70D4CED2DE3F000000000000F87FFFF6E43840FB983F000000000000F87F000000000000F87F000000000000F87FD8BC97DEBB9AEE3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F587F25EB29F3A03E000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F50D3389507A8EE3F8400EF211F93BB3F000000000000F87F000000000000F87F34507870DE42EC3F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F000000000000F87F32F30C9FE5A8EA3F000000000000F87F7A035AD21FC6EF3F86FB8C29E9F55A3E1E2A94ABB8C30F3F000000000000F87F000000000000F87F000000000000F87F22EAFC7BC70BB53F000000000000F87FFBCEBB8C93CDD13F000000000000F87F000000000000F87F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%2) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f64>) -> tensor<f64>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f64>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f64>, tensor<f64>) -> tensor<f64>
    "func.return"(%1) : (tensor<f64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

